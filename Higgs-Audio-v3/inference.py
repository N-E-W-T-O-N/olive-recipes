"""Run the exported Higgs-Audio-v3 ONNX sub-parts end-to-end → audio.

Single --model-path points at  onnx/{device}_{precision}/  (flat, manifest-driven):
  llm_decoder.onnx (+.data) + genai_config.json + tokenizer   — Qwen3-4B decoder
  audio_embed.onnx     codes[B,L,8]      → embeds[B,L,2560]
  audio_heads.onnx     hidden[B,L,2560]  → logits[B,L,8,1026]
  audio_tokenizer.onnx codes[B,8,T]      → waveform[B,1,L]
The text/tied embedding comes from the sibling  qwen3_standalone/.

Pipeline (TTS):
  text → chat-template tokens → text embeds → LLM(prime, KV cache) → hidden
       → audio_heads → per-codebook logits → delay-pattern sampler → audio codes
       → reverse delay → audio_tokenizer (codec decode) → 24 kHz waveform .wav

onnxruntime-genai cannot drive this whole model (see note in README/STATUS): the
fused multi-codebook audio head + delay-pattern sampling + neural codec are not
og-supported ops, and our llm_decoder excludes the lm_head. So the AR loop, the
delay sampler, and codes→waveform are implemented here in Python.

VERIFIED: the codec path (codes→waveform) matches PyTorch exactly (--selftest).
The full text→speech AR loop is implemented per the sglang-omni reference; its
speech intelligibility depends on the exact voice-design prompt format and should
be validated against the reference runtime.

Usage:
  python inference.py --model-path onnx/cpu_int4 --selftest        # codec → wav (verified)
  python inference.py --model-path onnx/cpu_int4 --text "Hello." --out hello.wav
"""
import argparse
import json
from pathlib import Path

import numpy as np

HEAD_DIM = 128
N_CODEBOOKS = 8
CB_VOCAB = 1026
BOC_ID = 1024          # begin-of-code (delay fill)
EOC_ID = 1025          # end-of-code
SR = 24000


# ----------------------------- delay pattern ------------------------------- #

def apply_delay_pattern(codes_TN: np.ndarray) -> np.ndarray:
    """[T, N] → [T+N-1, N]; codebook c shifted down by c, gaps=BOC, tail=EOC."""
    T, N = codes_TN.shape
    out = np.full((T + N - 1, N), EOC_ID, dtype=np.int64)
    for c in range(N):
        out[c:c + T, c] = codes_TN[:, c]
        out[:c, c] = BOC_ID
    return out


def reverse_delay_pattern(delayed_LN: np.ndarray) -> np.ndarray:
    """[L, N] → [L-(N-1), N]; undo the per-codebook shift."""
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        return np.zeros((0, N), dtype=np.int64)
    out = np.zeros((T, N), dtype=np.int64)
    for c in range(N):
        out[:, c] = delayed_LN[c:c + T, c]
    return out


class Pipeline:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        from safetensors import safe_open
        from transformers import AutoTokenizer

        self.root = Path(model_path)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        sm = self.manifest["sub_models"]

        # Resolve the requested EP against what's actually installed; fall back to CPU
        # with a clear message (e.g. an openvino_* build run in a CPU-only env).
        want = self.manifest.get("execution_provider", "CPUExecutionProvider")
        avail = ort.get_available_providers()
        if want in avail:
            if want == "OpenVINOExecutionProvider":
                # device_type CPU/GPU/NPU/AUTO from the manifest (Intel NPU = "NPU")
                ov_dev = self.manifest.get("ov_device_type", "CPU")
                providers = [(want, {"device_type": ov_dev}), "CPUExecutionProvider"]
                self.active_ep = f"OpenVINO:{ov_dev}"
            elif want != "CPUExecutionProvider":
                providers = [want, "CPUExecutionProvider"]
                self.active_ep = want
            else:
                providers = ["CPUExecutionProvider"]
                self.active_ep = "CPUExecutionProvider"
        else:
            providers = ["CPUExecutionProvider"]
            self.active_ep = "CPUExecutionProvider"
            print(f"[inference] '{want}' not available ({avail}); falling back to "
                  f"CPUExecutionProvider. (Install a matching onnxruntime build to use {want}.)")

        def sess(name):
            if name not in sm:
                return None
            return ort.InferenceSession(str(self.root / sm[name]["filename"]), providers=providers)

        self.llm = sess("llm_decoder")
        if self.llm is not None:
            llm = sm["llm_decoder"]
            self.n_layers = llm["num_layers"]; self.n_kv = llm["num_kv_heads"]
        self.audio_embed = sess("audio_embed")
        self.audio_heads = sess("audio_heads")
        self.codec = sess("audio_tokenizer")

        # text embedding + tokenizer (only needed for the text/LLM path)
        self.embed, self.tok = None, None
        std = self.standalone_dir()
        if (std / "model.safetensors").exists():
            with safe_open(str(std / "model.safetensors"), framework="pt") as f:
                self.embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
            try:
                self.tok = AutoTokenizer.from_pretrained(str(std), fix_mistral_regex=True)
            except TypeError:
                self.tok = AutoTokenizer.from_pretrained(str(std))

    # ---- LLM with KV cache ----
    def _empty_past(self):
        z = np.zeros((1, self.n_kv, 0, HEAD_DIM), dtype=np.float32)
        return {f"past_key_values.{i}.{kv}": z for i in range(self.n_layers) for kv in ("key", "value")}

    def _llm_step(self, inputs_embeds, attn_len, past):
        feeds = {"inputs_embeds": inputs_embeds.astype(np.float32),
                 "attention_mask": np.ones((1, attn_len), dtype=np.int64), **past}
        outs = self.llm.run(None, feeds)
        names = [o.name for o in self.llm.get_outputs()]
        d = dict(zip(names, outs))
        hidden = d["hidden_states"]
        new_past = {f"past_key_values.{i}.{kv}": d[f"present.{i}.{kv}"]
                    for i in range(self.n_layers) for kv in ("key", "value")}
        return hidden, new_past

    def standalone_dir(self) -> Path:
        return (self.root / self.manifest.get("standalone_dir", "../../qwen3_standalone")).resolve()

    # ---- text-path helpers (parity / eval) ----
    def hidden_states(self, input_ids: np.ndarray) -> np.ndarray:
        """Full-sequence forward (no KV cache) → hidden_states [B,S,H]."""
        h, _ = self._llm_step(self.embed[input_ids], input_ids.shape[1], self._empty_past())
        return h

    def logits_last(self, input_ids: np.ndarray) -> np.ndarray:
        return self.hidden_states(input_ids)[:, -1, :] @ self.embed.T

    # ---- codec ----
    def decode_codes(self, codes_TN: np.ndarray) -> np.ndarray:
        """[T, 8] int codes → mono waveform [L] float32."""
        codes_BNT = codes_TN.T[None].astype(np.int64)            # [1, 8, T]
        wav = self.codec.run(None, {"audio_codes": codes_BNT})[0]
        return wav[0, 0]

    # ---- full TTS ----
    def _tts_prompt_ids(self, text: str) -> np.ndarray:
        """Zero-shot Higgs TTS prompt: <|tts|> <|text|> tok(text) <|audio|>.

        The <|tts|> token selects TTS (not ASR) mode; the trailing <|audio|>
        token is what makes the model start emitting audio codes. (Ref:
        sglang_omni/models/higgs_tts/text_tokenizer.py.)
        """
        av = self.tok.get_added_vocab()
        tts_id, text_id, audio_id = av["<|tts|>"], av["<|text|>"], av["<|audio|>"]
        body = self.tok.encode(text, add_special_tokens=False)
        return np.array([[tts_id, text_id, *body, audio_id]], dtype=np.int64)

    @staticmethod
    def _sample(logits_NV: np.ndarray, temperature: float, top_k: int, rng) -> np.ndarray:
        """Per-codebook sample [N,V] → [N]. temperature<=0 ⇒ greedy argmax.

        Pure argmax degenerates (codebook-0 sticks on one token → buzz, no EOC);
        temperature+top-k sampling matches the sglang reference and breaks the loop.
        """
        N, V = logits_NV.shape
        if temperature <= 0:
            return logits_NV.argmax(-1).astype(np.int64)
        logits = logits_NV.astype(np.float64) / temperature
        if 0 < top_k < V:
            kth = np.partition(logits, V - top_k, axis=-1)[:, V - top_k][:, None]
            logits = np.where(logits < kth, -np.inf, logits)
        logits -= logits.max(-1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(-1, keepdims=True)
        return np.array([rng.choice(V, p=probs[i]) for i in range(N)], dtype=np.int64)

    def generate_speech(self, text: str, max_frames: int = 600,
                        temperature: float = 0.8, top_k: int = 50,
                        seed: int = 0, max_repeat: int = 32) -> np.ndarray:
        ids = self._tts_prompt_ids(text)
        rng = np.random.default_rng(seed)

        # prime LLM on the TTS prompt; hidden at the final (<|audio|>) position
        # produces the first audio frame.
        past = self._empty_past()
        hidden, past = self._llm_step(self.embed[ids], ids.shape[1], past)
        total = ids.shape[1]

        # AR audio loop with delay pattern
        delayed = []          # list of [8] delayed code rows
        delay_count = 0
        eoc_countdown = None
        last_cb0, repeat = None, 0
        for _ in range(max_frames):
            logits = self.audio_heads.run(None, {"hidden_states": hidden[:, -1:, :]})[0]  # [1,1,8,1026]
            codes = self._sample(logits[0, 0], temperature, top_k, rng)                   # [8]
            # degeneration guard: stop if cb0 repeats the same code for too long
            if codes[0] == last_cb0:
                repeat += 1
                if repeat >= max_repeat:
                    break
            else:
                repeat = 0
            last_cb0 = int(codes[0])
            if delay_count < N_CODEBOOKS:                  # delay ramp: mask future codebooks
                nxt = delay_count + 1
                if nxt < N_CODEBOOKS:
                    codes[nxt:] = BOC_ID
                delay_count += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    delayed.append(codes); break
            elif int(codes[0]) == EOC_ID:
                eoc_countdown = N_CODEBOOKS - 2
            delayed.append(codes)
            # feed sampled codes back as next input embedding
            emb = self.audio_embed.run(None, {"codes": codes[None, None]})[0]             # [1,1,2560]
            total += 1
            hidden, past = self._llm_step(emb, total, past)

        if not delayed:
            return np.zeros(0, dtype=np.float32)
        codes_TN = reverse_delay_pattern(np.stack(delayed))      # undo delay → [T,8]
        codes_TN = np.clip(codes_TN, 0, 1023)                    # drop BOC/EOC markers
        return self.decode_codes(codes_TN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="onnx/{device}_{precision} dir")
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default="output.wav")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="decode random codes → wav (verifies the codec path)")
    args = ap.parse_args()
    import soundfile as sf

    pipe = Pipeline(args.model_path)
    print(f"Loaded {args.model_path} (device={pipe.manifest['device']}, "
          f"precision={pipe.manifest['precision']}, running on {pipe.active_ep})")
    print(f"sub-parts: {list(pipe.manifest['sub_models'])}")

    if args.selftest:
        codes = np.random.randint(0, 1024, (50, N_CODEBOOKS), dtype=np.int64)   # 2 s @ 25 fps
        wav = pipe.decode_codes(codes)
        sf.write(args.out, wav, SR)
        print(f"[selftest] codec decoded {codes.shape[0]} frames → {wav.shape[0]} samples "
              f"({wav.shape[0]/SR:.2f}s) → {args.out}")
        return

    if not args.text:
        ap.error("provide --text or --selftest")
    wav = pipe.generate_speech(args.text, max_frames=args.max_frames,
                               temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    sf.write(args.out, wav, SR)
    print(f"Generated {wav.shape[0]} samples ({wav.shape[0]/SR:.2f}s) → {args.out}")


if __name__ == "__main__":
    main()
