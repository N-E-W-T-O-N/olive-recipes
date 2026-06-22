"""Run the exported Higgs-Audio-v3 ONNX sub-parts end-to-end → audio.

Single --model-path points at  onnx/{device}_{precision}/  (flat, manifest-driven):
  llm_decoder.onnx (+.data) + genai_config.json + tokenizer   — Qwen3-4B decoder
  audio_embed.onnx     codes[B,L,8]      → embeds[B,L,2560]
  audio_heads.onnx     hidden[B,L,2560]  → logits[B,L,8,1026]
  audio_tokenizer.onnx codes[B,8,T]      → waveform[B,1,L]   (codec decode)
  audio_encoder.onnx   wav[B,1,T]        → codes[B,8,frames] (codec encode; voice-clone ref)
  text_embed.onnx      input_ids[B,L]    → inputs_embeds[B,L,2560]  (token-embed Gather)
The token embedding is its own ONNX (text_embed) so the pipeline is self-contained —
no qwen3_standalone / original checkpoint needed at runtime (eval.py still uses it as a
PyTorch reference only).

Modes:
  • zero-shot TTS     : --text
  • voice clone       : --text + --ref-audio <wav> + --ref-text "<transcript>"
    Clone prompt (sglang ref): <|tts|> <|ref_text|> tok(ref_text) <|ref_audio|>
    [ref-audio codes, encoded by audio_encoder, delay-patterned, embedded by audio_embed]
    <|text|> tok(text) <|audio|>.

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
  # voice clone (reference audio + its transcript):
  python inference.py --model-path onnx/cpu_int4 --ref-audio ref.wav \
      --ref-text "exact transcript of ref.wav" --text "New sentence to say." --out clone.wav
"""
import argparse
import json
import sys
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
        prov = self.manifest.get("execution_provider", "CPUExecutionProvider")
        self.provider = prov

        def sess(name):
            if name not in sm:
                return None
            return ort.InferenceSession(str(self.root / sm[name]["filename"]), providers=[prov])

        self.llm = sess("llm_decoder")
        self.llm_dtype = np.float32
        if self.llm is not None:
            llm = sm["llm_decoder"]
            self.n_layers = llm["num_layers"]; self.n_kv = llm["num_kv_heads"]
            # CUDA/fp16 ModelBuilder decoders expect float16 inputs_embeds + KV cache;
            # CPU int4 expects float32. Hardcoding float32 fails on the CUDA build with
            # "Unexpected input data type ... expected float16". Read it from the graph.
            for inp in self.llm.get_inputs():
                if inp.name == "inputs_embeds":
                    self.llm_dtype = np.float16 if "float16" in inp.type else np.float32
        self.audio_embed = sess("audio_embed")
        self.audio_heads = sess("audio_heads")
        self.codec = sess("audio_tokenizer")
        self.audio_encoder = sess("audio_encoder")  # waveform → codes (voice-clone ref)
        self.text_embed = sess("text_embed")     # input_ids → inputs_embeds (Gather)

        # Token embedding: primary path is the standalone text_embed.onnx (a Gather),
        # so the pipeline is self-contained — no PyTorch model / qwen3_standalone needed
        # at runtime. self.embed (numpy table) is loaded ONLY if a standalone happens to
        # be present, purely for the eval-side logits_last() (hidden @ embedᵀ).
        self.embed, self.tok = None, None
        std = self.standalone_dir()
        if self.text_embed is None and std and (std / "model.safetensors").exists():
            with safe_open(str(std / "model.safetensors"), framework="pt") as f:
                self.embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
        elif std and (std / "model.safetensors").exists():
            try:
                with safe_open(str(std / "model.safetensors"), framework="pt") as f:
                    self.embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
            except Exception:
                pass
        # tokenizer: model dir ships tokenizer.json alongside the ONNX parts.
        for cand in [self.root, std]:
            if cand and (cand / "tokenizer.json").exists():
                try:
                    self.tok = AutoTokenizer.from_pretrained(str(cand), fix_mistral_regex=True)
                except TypeError:
                    self.tok = AutoTokenizer.from_pretrained(str(cand))
                break

    @staticmethod
    def _run(sess, feeds):
        """Run a session, casting each float feed to the dtype the graph expects
        (fp16 builds need float16; int4/cpu need float32). int inputs pass through."""
        want = {i.name: i.type for i in sess.get_inputs()}
        cast = {}
        for k, v in feeds.items():
            t = want.get(k, "")
            if "float16" in t:
                cast[k] = np.asarray(v, np.float16)
            elif "float" in t:
                cast[k] = np.asarray(v, np.float32)
            else:
                cast[k] = v
        return sess.run(None, cast)

    # ---- LLM with KV cache ----
    def _empty_past(self):
        z = np.zeros((1, self.n_kv, 0, HEAD_DIM), dtype=self.llm_dtype)
        return {f"past_key_values.{i}.{kv}": z for i in range(self.n_layers) for kv in ("key", "value")}

    def _llm_step(self, inputs_embeds, attn_len, past):
        feeds = {"inputs_embeds": inputs_embeds.astype(self.llm_dtype),
                 "attention_mask": np.ones((1, attn_len), dtype=np.int64), **past}
        outs = self.llm.run(None, feeds)
        names = [o.name for o in self.llm.get_outputs()]
        d = dict(zip(names, outs))
        hidden = np.asarray(d["hidden_states"], np.float32)   # downstream math in fp32
        new_past = {f"past_key_values.{i}.{kv}": d[f"present.{i}.{kv}"]
                    for i in range(self.n_layers) for kv in ("key", "value")}
        return hidden, new_past

    def standalone_dir(self):
        """Locate the extracted Qwen3 dir (model.embed_tokens.weight + tokenizer).

        The manifest's relative `standalone_dir` assumes onnx/{dev}_{prec}/ nesting;
        flat layouts (model dir == cpu_int4/) break that path. Search several
        candidates and return the first that actually holds model.safetensors, else
        the first that holds a tokenizer, else None.
        """
        cands = [
            self.root / "qwen3_standalone",
            self.root / self.manifest.get("standalone_dir", "../../qwen3_standalone"),
            self.root,
        ]
        cands = [c.resolve() for c in cands]
        for c in cands:
            if (c / "model.safetensors").exists():
                return c
        for c in cands:
            if (c / "tokenizer.json").exists():
                return c
        return None

    # ---- token embedding (self-contained: text_embed.onnx Gather) ----
    def embed_ids(self, input_ids: np.ndarray) -> np.ndarray:
        """input_ids [B,L] → inputs_embeds [B,L,H] via text_embed.onnx (preferred),
        falling back to the numpy table if only a standalone is present."""
        if self.text_embed is not None:
            return self.text_embed.run(None, {"input_ids": np.asarray(input_ids, np.int64)})[0]
        if self.embed is not None:
            return self.embed[input_ids]
        raise RuntimeError(
            "No token embedding available: ship text_embed.onnx in the model dir "
            "(run _build_text_embed.py) or provide a qwen3_standalone/model.safetensors.")

    # ---- text-path helpers (parity / eval) ----
    def hidden_states(self, input_ids: np.ndarray) -> np.ndarray:
        """Full-sequence forward (no KV cache) → hidden_states [B,S,H]."""
        h, _ = self._llm_step(self.embed_ids(input_ids), input_ids.shape[1], self._empty_past())
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
    def _sample(logits_NV: np.ndarray, temperature: float, top_k: int, top_p: float,
                rng) -> np.ndarray:
        """Per-codebook sample [N,V] → [N]. temperature<=0 ⇒ greedy argmax.

        temperature + top-k + nucleus(top_p) sampling (matches the sglang reference).
        Pure argmax degenerates (codebook-0 sticks → buzz, no EOC).
        """
        N, V = logits_NV.shape
        if not np.isfinite(logits_NV).all():
            raise RuntimeError(
                "audio_heads produced non-finite (NaN/Inf) logits — the llm_decoder almost "
                "certainly computed garbage upstream (broken INT4 contrib kernel, e.g. "
                "GroupQueryAttention on a prebuilt ARM64/Jetson onnxruntime-gpu wheel). "
                "Use a source-built ORT with CUDA contrib kernels for your SM, or the CPU EP. "
                "See STATUS.md 'CUDA int4 / ARM'.")
        if temperature <= 0:
            return logits_NV.argmax(-1).astype(np.int64)
        logits = logits_NV.astype(np.float64) / temperature
        if 0 < top_k < V:
            kth = np.partition(logits, V - top_k, axis=-1)[:, V - top_k][:, None]
            logits = np.where(logits < kth, -np.inf, logits)
        logits -= logits.max(-1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(-1, keepdims=True)
        if 0 < top_p < 1.0:                                   # nucleus filter per codebook
            order = np.argsort(probs, axis=-1)[:, ::-1]
            sorted_p = np.take_along_axis(probs, order, axis=-1)
            csum = np.cumsum(sorted_p, axis=-1)
            keep = csum - sorted_p < top_p                    # keep until cumulative ≥ top_p
            sorted_p = np.where(keep, sorted_p, 0.0)
            probs = np.zeros_like(probs)
            np.put_along_axis(probs, order, sorted_p, axis=-1)
            probs /= probs.sum(-1, keepdims=True)
        return np.array([rng.choice(V, p=probs[i]) for i in range(N)], dtype=np.int64)

    def _check_text_ready(self):
        if self.tok is None:
            raise RuntimeError(f"Tokenizer not found (tokenizer.json) in {self.root}.")
        if self.text_embed is None and self.embed is None:
            raise RuntimeError(
                "Token embedding not found. Ship text_embed.onnx in the model dir "
                "(python optimize.py --components text_embed). --selftest works without it.")

    @staticmethod
    def _load_wav(path: str) -> np.ndarray:
        """Read a wav → mono float32 @ 24 kHz."""
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)

    def encode_audio(self, wav: np.ndarray) -> np.ndarray:
        """Reference waveform (mono 24 kHz) → codes [frames, 8] via audio_encoder.onnx."""
        if self.audio_encoder is None:
            raise RuntimeError(
                "audio_encoder.onnx not found — needed to encode reference audio for voice "
                "cloning. Build it: `python optimize.py --components audio_encoder`.")
        x = np.asarray(wav, np.float32).reshape(-1)
        # The traced graph folded the acoustic/semantic length-alignment branch
        # (modeling:548), so it only works when both encoder streams come out the same
        # length — which happens iff the input is a whole number of codec frames
        # (hop = SR/fps = 960 samples). Arbitrary lengths fail with a Concat mismatch
        # (e.g. 1112 vs 1111). Pad up to the next multiple of HOP with silence.
        HOP = 960                                                       # SR/fps = 24000/25
        n = x.shape[0]
        pad_to = max(HOP, ((n + HOP - 1) // HOP) * HOP)
        if pad_to != n:
            x = np.pad(x, (0, pad_to - n))
        x = x.reshape(1, 1, -1)
        codes = self._run(self.audio_encoder, {"input_values": x})[0]   # [1,8,frames]
        return codes[0].T.astype(np.int64)                              # [frames,8]

    def _run_ar(self, prefill_embeds: np.ndarray, max_frames: int, temperature: float,
                top_k: int, top_p: float, seed: int, max_repeat: int) -> np.ndarray:
        """Prime the LLM on prefill_embeds [1,S,2560], then AR-generate audio codes with
        the delay pattern. Shared by zero-shot and voice-clone. Returns waveform [L]."""
        rng = np.random.default_rng(seed)
        past = self._empty_past()
        hidden, past = self._llm_step(prefill_embeds, prefill_embeds.shape[1], past)
        total = prefill_embeds.shape[1]

        # Silent-zeros guard: on some runtimes (notably ARM64/Jetson CUDA builds) the
        # llm_decoder's GroupQueryAttention / MatMulNBits contrib kernels can register
        # yet compute all-zeros — producing silent/garbage audio with no error. Catch it
        # here instead of after minutes of generation. (See STATUS.md "CUDA int4 / ARM".)
        if float(np.abs(hidden).mean()) < 1e-8:
            raise RuntimeError(
                f"llm_decoder produced all-zero hidden states on provider '{self.provider}' "
                "— the INT4 contrib kernels (GroupQueryAttention / MatMulNBits) are not "
                "computing on this runtime/hardware. Known on ARM64/Jetson CUDA builds. "
                "Use the CPU provider, an fp16 build, or a source-built onnxruntime with the "
                "CUDA contrib kernels for your GPU arch. See STATUS.md.")

        delayed = []; delay_count = 0; eoc_countdown = None
        last_cb0, repeat = None, 0
        stop = "cap"          # how the loop ended: cap | eoc | repeat
        for _ in range(max_frames):
            logits = self._run(self.audio_heads, {"hidden_states": hidden[:, -1:, :]})[0]  # [1,1,8,1026]
            codes = self._sample(logits[0, 0], temperature, top_k, top_p, rng)            # [8]
            if codes[0] == last_cb0:
                repeat += 1
                if repeat >= max_repeat:
                    stop = "repeat"; break
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
                    delayed.append(codes); stop = "eoc"; break
            elif int(codes[0]) == EOC_ID:
                eoc_countdown = N_CODEBOOKS - 2
            delayed.append(codes)
            emb = self.audio_embed.run(None, {"codes": codes[None, None]})[0]             # [1,1,2560]
            total += 1
            hidden, past = self._llm_step(emb, total, past)

        secs = len(delayed) * 960 / SR
        if stop == "cap":
            print(f"  [warning] hit --max-frames={max_frames} ({secs:.1f}s) — output likely "
                  f"TRUNCATED mid-speech. Re-run with a larger --max-frames (≈ seconds × 25).",
                  file=sys.stderr)
        elif stop == "repeat":
            print(f"  [note] stopped early at {secs:.1f}s by the repeat guard "
                  f"(codebook-0 repeated {max_repeat}×). Try adjusting --temperature/--top-k.",
                  file=sys.stderr)
        else:
            print(f"  [ok] natural end-of-speech (EOC) at {secs:.1f}s.", file=sys.stderr)

        # On a repeat-stop the trailing ~max_repeat frames ARE the degenerate buzz that
        # tripped the guard — drop them so the audio doesn't end in garbage.
        if stop == "repeat" and len(delayed) > max_repeat:
            del delayed[-max_repeat:]
        if not delayed:
            return np.zeros(0, dtype=np.float32)
        codes_TN = reverse_delay_pattern(np.stack(delayed))      # undo delay → [T,8]
        codes_TN = np.clip(codes_TN, 0, 1023)                    # drop BOC/EOC markers
        return self.decode_codes(codes_TN)

    def build_prefill(self, text: str, ref_audio: str = None, ref_text: str = None) -> np.ndarray:
        """Assemble the prefill embeds [1,S,2560] for zero-shot or voice-clone.
        Zero-shot: <|tts|> <|text|> tok(text) <|audio|>.
        Clone:     <|tts|> <|ref_text|> tok(ref) <|ref_audio|> [ref-code embeds] <|text|> tok(text) <|audio|>."""
        self._check_text_ready()
        if not ref_audio:
            return self.embed_ids(self._tts_prompt_ids(text))
        if self.audio_encoder is None or self.audio_embed is None:
            raise RuntimeError("voice clone needs audio_encoder.onnx + audio_embed.onnx in the model dir.")
        if not ref_text:
            raise RuntimeError("voice clone requires ref_text (transcript of the reference audio).")
        av = self.tok.get_added_vocab()
        ref_codes = self.encode_audio(self._load_wav(ref_audio))         # [T,8]
        delayed_ref = apply_delay_pattern(ref_codes)                     # [T+N-1,8]
        ref_embeds = self._run(self.audio_embed, {"codes": delayed_ref[None].astype(np.int64)})[0]

        def seg(ids):
            return self.embed_ids(np.asarray([ids], np.int64))
        ref_text_toks = self.tok.encode(ref_text, add_special_tokens=False)
        body = self.tok.encode(text, add_special_tokens=False)
        prefill = np.concatenate([
            seg([av["<|tts|>"]]),
            seg([av["<|ref_text|>"], *ref_text_toks]),
            seg([av["<|ref_audio|>"]]),
            ref_embeds.astype(np.float32),
            seg([av["<|text|>"], *body]).astype(np.float32),
            seg([av["<|audio|>"]]).astype(np.float32),
        ], axis=1)
        print(f"  [clone] ref {ref_codes.shape[0]} frames + ref_text {len(ref_text_toks)} toks "
              f"+ text {len(body)} toks → prefill {prefill.shape[1]}", file=sys.stderr)
        return prefill

    def generate_speech(self, text: str, max_frames: int = 2000, temperature: float = 0.8,
                        top_k: int = 50, top_p: float = 1.0, seed: int = 0,
                        max_repeat: int = 32) -> np.ndarray:
        """Zero-shot TTS."""
        return self._run_ar(self.build_prefill(text), max_frames, temperature, top_k, top_p,
                            seed, max_repeat)

    def generate_clone(self, text: str, ref_audio: str, ref_text: str, max_frames: int = 2000,
                       temperature: float = 0.8, top_k: int = 50, top_p: float = 1.0,
                       seed: int = 0, max_repeat: int = 32) -> np.ndarray:
        """Voice clone from reference audio + its transcript."""
        prefill = self.build_prefill(text, ref_audio, ref_text)
        return self._run_ar(prefill, max_frames, temperature, top_k, top_p, seed, max_repeat)


def main():
    ap = argparse.ArgumentParser(
        description="Higgs-Audio-v3 ONNX text→speech (self-contained: no original model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)   # shows defaults in --help
    ap.add_argument("--model-path", required=True, help="onnx/{device}_{precision} dir (manifest-driven)")
    ap.add_argument("--text", default=None, help="text to synthesize (omit with --selftest)")
    ap.add_argument("--ref-audio", default=None,
                    help="reference wav for VOICE CLONING (mono, any sr — resampled to 24k). "
                         "Requires --ref-text and audio_encoder.onnx in the model dir")
    ap.add_argument("--ref-text", default=None,
                    help="transcript of --ref-audio (required for voice cloning)")
    ap.add_argument("--out", default="output.wav", help="output wav path")
    ap.add_argument("--max-frames", type=int, default=2000,
                    help="max audio frames @ 25 fps (≈ seconds × 25; ~80 s). Raise for long "
                         "text — a 'hit --max-frames' warning means the audio was truncated")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="per-codebook sampling temperature (0 = greedy argmax)")
    ap.add_argument("--top-k", type=int, default=50, help="top-k sampling cutoff per codebook")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus sampling (1.0=off; try ~0.95 to curb tail degeneration)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    ap.add_argument("--selftest", action="store_true",
                    help="decode random codes → wav (verifies the codec path only)")
    args = ap.parse_args()
    import soundfile as sf

    pipe = Pipeline(args.model_path)
    print(f"Loaded {args.model_path} (device={pipe.manifest['device']}, precision={pipe.manifest['precision']})")
    print(f"sub-parts: {list(pipe.manifest['sub_models'])}")

    if args.selftest:
        codes = np.random.randint(0, 1024, (50, N_CODEBOOKS), dtype=np.int64)   # 2 s @ 25 fps
        wav = pipe.decode_codes(codes)
        sf.write(args.out, wav, SR)
        print(f"[selftest] codec decoded {codes.shape[0]} frames → {wav.shape[0]} samples "
              f"({wav.shape[0]/SR:.2f}s) → {args.out}")
        return

    if not args.text:
        ap.error("provide --text (optionally with --ref-audio/--ref-text for cloning) or --selftest")
    if args.ref_audio:                          # voice clone
        if not args.ref_text:
            ap.error("--ref-audio requires --ref-text (transcript of the reference audio)")
        wav = pipe.generate_clone(args.text, args.ref_audio, args.ref_text,
                                  max_frames=args.max_frames, temperature=args.temperature,
                                  top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    else:                                        # zero-shot
        wav = pipe.generate_speech(args.text, max_frames=args.max_frames,
                                   temperature=args.temperature, top_k=args.top_k,
                                   top_p=args.top_p, seed=args.seed)
    sf.write(args.out, wav, SR)
    print(f"Generated {wav.shape[0]} samples ({wav.shape[0]/SR:.2f}s) → {args.out}")


if __name__ == "__main__":
    main()
