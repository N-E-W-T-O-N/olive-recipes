# VibeVoice → ONNX — Engineer's Handoff Note

_Written by the departing engineer. Everything here was learned the hard way; read it before
touching the code. Companion files: `STATUS.md` (running log), `.claude/skills/vibevoice-onnx/`
(the operational skill), and the checklist at the bottom of this note._

---

## 1. The one-paragraph mental model

The VibeVoice family is **one Qwen2 LLM backbone + audio VAE tokenizers + small glue MLPs**, in
four checkpoints that *look* alike but differ in critical details. We do NOT export the composite
model. We decompose each checkpoint into standalone ONNX sub-parts — LLM decoder via
onnxruntime-genai **ModelBuilder** (int4/fp16, KV cache), everything else via **Olive**
(dynamo exporter) — and re-compose them at inference time in plain Python + onnxruntime
(`common.py`). Every exported part is parity-checked against PyTorch (cosine ~1.0) before we
trust it. That decomposition + parity discipline is the whole method.

## 2. The four checkpoints — what actually differs (memorize this table)

| Key | HF layout | LLM | lm_head? | Acoustic tokenizer naming | Loads via |
|---|---|---|---|---|---|
| `1.5b` | codes/ (`vibevoice`) | Qwen2.5-1.5B, 28L | **No** (head = diffusion) | `encoder.downsample_layers.*` | `codes/` |
| `asr` | codes/ (`vibevoice`, arch `VibeVoiceForASRTraining`) | Qwen2.5-**7B** | **Yes** (top-level `lm_head.weight`) | same as 1.5b | `codes/` |
| `asr-hf` | transformers-native (`vibevoice_asr`) | Qwen2.5-**7B** | **Yes** (`language_model.lm_head.*`) | `acoustic_tokenizer_encoder.conv_layers.*` | transformers |
| `realtime` | codes/ (`vibevoice_streaming`) | Qwen2.5-0.5B, **20L (config says 24!)** | No | `decoder.stages.*` — **decoder-only, no encoder shipped** | `codes/` |

Rules that fall out of this:
- **Never assume tokenizer classes are interchangeable across checkpoints.** Same class name,
  three different weight namings. Always load with `strict=False` and **assert 0 missing /
  0 unexpected** — that assert is the tripwire that caught every mismatch.
- **TTS backbones** build with `exclude_embeds + exclude_lm_head` (inputs_embeds → hidden_states);
  **ASR decoders keep lm_head** (→ logits). `optimize.py`'s `MODELS` registry encodes this.
- `asr` vs `asr-hf` are **different models** with different key layouts — resist merging them.
- Realtime has **TWO embed tables** (`language_model` + `tts_language_model`) that differ.
  `common.EMBED_KEY` maps each key to the exact table matching the exported decoder. A
  first-match heuristic here silently produced garbage audio once. Don't reintroduce it.

## 3. Traps that cost real time (do not rediscover these)

1. **`import vibevoice` collides with transformers** (both register the same model_type) and the
   package `__init__` pulls diffusers + a qwen2-tokenizer module renamed in transformers ≥5.10.
   Fix: `_codes_import()` in `user_script.py` — inject empty namespace packages so only the one
   target module is imported, and shim `Auto*.register` to swallow duplicate registrations.
2. **dynamo ignores `dynamic_axes`.** Use `dynamic_shapes` in io_configs or the dim is baked
   static. Also: the frame axis of `latents[B, frames, 64]` is **dim 1** (dim 2 is vae_dim) —
   we shipped that off-by-one once.
3. **Timesteps must be float32.** The diffusion head's `TimestepEmbedder` casts its sinusoidal
   embedding back to `t.dtype` before a float MLP; int64 timesteps crash the matmul.
4. **CFG needs ONE sample.** Conditional and unconditional eps must come from the SAME noisy
   latent (duplicate the sample for the head call only, each step). Two independent samples =
   guidance mixed with a noise difference = degraded audio, no error.
5. **int4 is fine for token generation, lossy for raw hidden states** (first-token hidden cos
   ~0.83 vs fp32 across 28 layers of 4-bit RTN). The TTS diffusion head is conditioned on hidden
   states → **use fp16 builds for audio quality**; int4 for footprint experiments only.
6. **7B/8B int4 ModelBuilder serialize needs ~16–20 GB free RAM** (chandra-8B and both ASR 7Bs
   hit this). Extraction is separate and cheap — stream shards, `del` after each. Subprocess
   isolation does NOT help a single large serialize; you need physical RAM.
7. **Parity metric footnote:** cosine collapses on near-silent outputs (decoder fed random
   latents ≈ silence → cosine noise-dominated while max|Δ| ≈ 1e-10). Pass criterion is
   `cos ≥ 0.99 OR max|Δ| tiny` — that's deliberate, not sloppy.
8. **Windows consoles are cp1252.** Olive logs emoji → `UnicodeEncodeError`. Every entrypoint
   reconfigures stdout/stderr to UTF-8 at import. Keep that block.
9. **`argparse nargs='*'` + trailing positional don't mix** — `--components` is comma-separated
   for that reason.
10. **VibeVoice repos ship no tokenizer/processor.** We fetch the plain Qwen2.5 tokenizer; exact
    prompt layout and voice-cloning need the original processor assets (see codes/ `processor/`).
11. **fp16 conversion needs an `op_block_list` for shape/resample ops.** `OnnxFloatToFloat16` on the
    VAE/conv graphs will emit an invalid graph if it touches `ConstantOfShape` (its fp16 output hits a
    float32 consumer → `Type Error … does not match expected type (tensor(float))`), and `ConvTranspose`
    /`Resize`/`Range` gain nothing from fp16. `build_olive` blocks `["ConstantOfShape","ConvTranspose",
    "Resize","Range"]`. (The standalone `acoustic` encoder is where this first bit — the codes/-layout
    1.5b acoustic encoder happened not to.)
12. **int4 only quantizes the LLM.** Keys with no LLM (`acoustic`) or the audio/VAE/DiT components have
    no MatMulNBits-quantizable weights, so int4 is a no-op that silently re-emits fp32. `build_model`
    warns and downgrades `int4→fp32` for LLM-less keys so you don't ship a misnamed fp32 copy.
13. **Eval must feed each ONNX its DECLARED input dtype.** fp16 builds expect `float16`; feeding fp32
    dummies → `Unexpected input data type`. `eval.py` (`parity_component` + `whole_pipeline._feed`) casts
    per graph; float32-pinned inputs (diffusion `timesteps`) stay fp32. Without this, fp16 builds falsely
    "fail" eval though the graphs are fine.
14. **The VibeVoice source is VENDORED at `VibeVoice/vibevoice/` (~591 KB, MIT — see `VIBEVOICE_LICENSE`).**
    NOT a submodule, NOT a pip/git dependency. `_vibevoice_dir()` (user_script + common) returns that
    directory and raises if it's missing. The former `codes/` submodule, the `vibevoice-repo/` workspace
    clone (~263 MB each), the `[submodule]` in `.gitmodules`, the `vibevoice @ git+…` line in the onnx
    pyprojects, and the root-pyproject `vibevoice` dep + `vibevoice-repo` workspace were ALL removed so
    the tree is self-contained and uploadable. Imports still go through the isolated-import shim
    (`_codes_import` / the namespace-injection in `make_scheduler`/`_load_vv_processor`) — see trap 1;
    never `import vibevoice`. Only the required subtree is vendored (modular/, processor/, schedule/,
    scripts/, configs/ — no demo/, finetuning-asr/, vllm_plugin/).
15. **The 1.5B has a learned EOS — do NOT ship a fixed frame budget.** `lm_head` is TIED to
    `embed_tokens` (`_tied_weights_keys=["lm_head.weight"]`), so logits = `hidden @ embedᵀ` with NO
    lm_head export. VibeVoice reuses vision tokens for speech: stop when the model predicts speech-end
    (`<|vision_end|>`) or `<|endoftext|>`. `inference.py` computes this each frame via
    `common.load_embed_matrix()` and breaks. This killed the "jargon tail" (a real sonnet stopped at
    frame 101 of a 172 cap → 13.5 s not 22.9 s). `--max-frames` is now just a safety cap. Upstream
    ships no non-streaming 1.5B `generate` (codes/ @ 303b283 = training forward + streaming only);
    single-shot AR + this EOS is our reconstruction and it works — chunking regressed it, do not reintroduce.
    (Upstream check was against the vendored `vibevoice/modular/modeling_vibevoice.py` @ upstream 303b283.)

## 4. Architecture of our code (7 files, one direction of dependency)

```
optimize.py      MODELS registry (per-key: extract fn, exclude flags, olive component specs)
   │             + ensure_checkpoint (auto-download) + detect_model_type + output layout
   ▼
user_script.py   ALL loaders. codes/ isolated-import shim; per-checkpoint weight collectors;
                 4× extract_qwen2_* (standalone HF dirs for ModelBuilder); io_configs; dummies.
   ▼
common.py        Inference primitives: OnnxLLM (manual KV-cache driver over the genai graph,
                 hidden_states OR logits), DiffusionSampler (DPM + CFG), audio io, EMBED_KEY,
                 resolve_from_path (parses onnx/{key}/{device}_{precision}).
   ▼
inference.py / inference_asr.py / inference_realtime.py   thin drivers (one positional: built dir)
eval.py          parity (A) + whole-pipeline TTS checks (B); reuses the registry.
```

Conventions: output layout `onnx/{key}/{device}_{precision}`; model keyword is always the FINAL
positional; drivers take ONLY the built dir and derive model_id/device/precision from the path;
`codes/` = unmodified `github.com/microsoft/VibeVoice` @ `303b283` (submodule-able, imported
read-only by path — never pip-install it, never edit it).

## 5. What is done vs pending

**Done & parity-verified (cos ~1.0):** 19 sub-models — 1.5b (7/7 incl. int4 LLM), realtime (4/4),
asr front-end (5), asr-hf front-end (3). End-to-end TTS runs for 1.5b + realtime.

**Pending:** (a) 7B ASR LLM builds — RAM-bound, run on a big-memory machine; front-end +
`inference_asr.py` are ready and degrade cleanly without it. (b) fp16 builds for TTS quality.
(c) Encoders bake audio length at 24000 samples (switch io_configs to `dynamic_shapes` to lift).
(d) Learned EOS (1.5b lm_head / realtime `tts_eos_classifier`) not exported → `--max-frames`
stop. (e) Cleanup: fold the 4 extractors into one helper; dedupe the ~9 shard-collect loops;
delete dead stub classes; factor the duplicated driver frame-loop into `common.py`.

**Environment:** use the PROJECT env (`uv run` from the repo; deps in root `pyproject.toml`).
Do NOT add PEP-723 inline headers — they bypass the curated venv. CUDA needs GPU builds of
onnxruntime/genai present in the env. `.gitmodules` needs fixing: root-level,
`path = VibeVoice/codes`, url `https://github.com/microsoft/VibeVoice`, pin `303b283`.

---

# THE CHECKLIST — adding a new VibeVoice-family checkpoint (or re-running everything)

## A. Recon (30 min — do not skip)
- [ ] Dump `config.json`: `model_type`, `architectures`, sub-configs (`decoder_config` /
      `text_config`, `*_tokenizer_config`, `diffusion_head_config` — note `hidden_size`!).
- [ ] Dump weight groups: `Counter('.'.join(k.split('.')[:2]) for k in index/weight_map)`.
      Identify: LLM prefix, lm_head (present? where?), tokenizer groups + their naming style,
      connectors, heads. Compare against the table in §2.
- [ ] Decide codes/ vs transformers-native per component by **instantiating and loading with
      strict=False; require 0 missing / 0 unexpected**. Try both if unsure.
- [ ] Check actual layer count vs config (`realtime` lied: 20 real vs 24 configured).

## B. Wire loaders (`user_script.py`)
- [ ] Weight collector per component (prefix-strip; stream shards for >2 GB; assert 0/0).
- [ ] Extractor for the LLM → standalone Qwen2 dir (correct config sub-key, tokenizer id,
      lm_head kept iff the model emits text; override layer count if config lies).
- [ ] io_config: `dynamic_shapes` (NOT `dynamic_axes`), frame axis = dim 1, float32 timesteps.
- [ ] Register in `optimize.py` `MODELS` (+ `HF_REPO`, `common.EMBED_KEY`, `common.MODEL_IDS`).

## C. Build & verify (never skip verify)
- [ ] `uv run optimize.py --device cuda --precision fp16 <key>` (int4 only for the LLM footprint;
      `--exclude-llm` if RAM-bound). Output: `onnx/<key>/<device>_<precision>/`.
- [ ] `uv run eval.py <key>` → every component PASS; whole-pipeline codec round-trip
      corr > 0.99 / SNR > +15 dB on the tone test.
- [ ] LLM structural check: MatMulNBits/GQA counts = layer count; inputs = inputs_embeds (TTS)
      or logits output (ASR). KV-driver sanity: incremental step == full prefill (cos 1.0).
- [ ] Smoke the matching driver end-to-end (`--max-frames 8` is enough to prove the loop).

## D. Ship
- [ ] Update `STATUS.md` matrix row + a dated section (what, parity numbers, gotchas hit).
- [ ] No scratch files (`_*.py`, `*.log`), no stale `qwen2_*_standalone/` (15 GB each!), no
      old-layout outputs. Disk-full here corrupts builds mid-serialize.
- [ ] If it's a new trap, add it to §3 of this note. That's the contract.

# THE 5-MINUTE OPERATOR CARD (for someone who just wants to run it)

```bash
# build (downloads checkpoint if absent; output → onnx/<key>/<device>_<precision>)
uv run optimize.py --device cuda --precision fp16 1.5b        # or asr | asr-hf | realtime | all
uv run optimize.py --exclude-llm asr                          # front-end only (low RAM)

# verify
uv run eval.py 1.5b

# run
uv run inference.py          --text "Hello world." onnx/1.5b/cuda_fp16
uv run inference_realtime.py --text "Hi."          onnx/realtime/cuda_fp16
uv run inference_asr.py      --audio speech.wav    onnx/asr-hf/cuda_fp16
```
Drivers print `model_id / device / precision` derived from the path and refuse the wrong model
type. If ASR says the 7B LLM isn't built, build it on a ≥32 GB-free-RAM machine.
