# Qwen3.8-27B → ONNX — Handoff Note

_Scaffolded 2026-09-28. Recon done, folder structure built, NOTHING has been exported or
run yet — no weights downloaded, no Olive pipeline executed, no eval performed._

## 1. What this checkpoint is

`Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`) is a **VLM in the
same hybrid architecture family already converted twice in this repo** — `Surya-2/` and
`chandra-ocr-2/builtin/` (see the repo-wide `olive-model-recipes` skill's project index) — plus
`Qwen-Qwen3.5-0.8B/builtin/`, which is this project's direct structural template. It is simply
much bigger (27B vs their smaller sizes) and has one component none of those three have: an MTP
head (see §3).

**Weight footprint: 55.6 GB (bf16), 1199 tensors, 18 safetensors shards.** Only the small recon
files (config/tokenizer/processor JSON, ~22 MB total) have been downloaded so far — the actual
`--include "*.safetensors"` download only happens implicitly the first time `optimize.py` runs
Olive against `Qwen/Qwen3.8-27B` (via `HfModel`/`hf_hub_download` inside `user_script.py`'s
`_load_base_model`). Budget disk + time for that before the first real build attempt.

## 2. Architecture (verified against actual weights, not just config.json — trap #8)

- **Text decoder** (`text_config`, `qwen3_5_text`): 64 layers (verified 64/64 against the
  weight index — config does NOT lie here, unlike VibeVoice-realtime), hidden 5120,
  intermediate 17408. **Hybrid attention**: `full_attention_interval: 4` — every 4th layer
  (indices 3, 7, 11, …) is standard GQA full attention (24 heads / 4 kv heads, head_dim 256,
  partial RoPE 0.25, QK-norm); the rest are **linear attention** (GatedDeltaNet-style:
  `conv1d`, `in_proj_{a,b,qkv,z}`, `A_log`, `dt_bias`, `norm`, `out_proj` — same mechanism as
  the three existing qwen3_5 recipes). **MROPE**: `mrope_section=[11,11,10]`, interleaved.
  `lm_head` is **untied** (`tie_word_embeddings: False`) — unlike Qwen3.5-0.8B, which IS tied
  (its `text.json` has a `TieWordEmbeddings` surgery; **ours must NOT** — already omitted from
  every `text.json` in this scaffold, don't add it back).
- **Vision tower**: SigLIP-style, depth 27 (verified 27/27), hidden 1152, `out_hidden_size=5120`
  (== text hidden_size — used in `user_script.py`'s embedding dummy inputs), patch 16,
  spatial_merge 2, temporal_patch 2. Standard `Qwen3VLProcessor` /
  `Qwen2VLImageProcessorFast` / `Qwen3VLVideoProcessor` (video input supported too).
- **MTP head** (`mtp.*`, 14 tensors, `mtp_num_hidden_layers: 1`): a Multi-Token-Prediction /
  speculative-decoding auxiliary head. **Not present in any of the 3 existing qwen3_5 recipes'
  vendored `codes/modeling_qwen3_5.py`.** Decision for this first pass: **skip exporting it.**
  Standard MTP heads are opt-in generation acceleration, not required for correct
  single-token-at-a-time output via `lm_head` — dropping it should not affect correctness, only
  forgo speculative-decode speedup. Revisit if that speedup turns out to matter.
- **Tokenizer**: `Qwen2Tokenizer`, vocab 248320. `eos_token_id` in `generation_config.json` is
  `[248046, 248044]` — **two valid stop ids**, both wired into every `optimize.py`'s
  `update_genai_config` (`config["model"]["eos_token_id"] = [248046, 248044]`). No landmines
  like VibeVoice's reused-token markers; token ids for bos/eos/pad/image/video/vision_start are
  all clean, normal values (248044/248046/248044/248056/248057/248053).

## 3. Folder structure — deliberately NOT this repo's default shape

The repo-wide convention (see `olive-model-recipes` skill) prefers a single Python-driven
`optimize.py` that generates Olive configs in-Python, parameterized by `--device`/`--precision`
(the `chandra/builtin` shape) — **not** static per-target JSON files (the `Qwen-Qwen3.5-0.8B`
shape). **This project explicitly overrides that preference per user instruction**: 4
self-contained device/precision folders, each with its own Olive JSON configs AND its own
`eval.py`/`inference.py` (even the existing `Qwen-Qwen3.5-0.8B/builtin` precedent only
duplicates the JSON configs per folder — it shares one `eval.py`/`inference.py` across all
targets. This project duplicates those too, per explicit request):

```
Qwen-Qwen3.8-27B/
├── codes/modeling_qwen3_5.py   # vendored, COPIED VERBATIM from Qwen-Qwen3.5-0.8B/builtin/ —
│                                 fully config-driven (Qwen3_5Config.from_pretrained), no
│                                 hardcoded 0.8B dims found; should work unmodified for 27B.
├── user_script.py               # SHARED loaders (embedding + vision only — text decoder goes
│                                 through ModelBuilder's HfModel path, no custom loader needed)
├── optimize.py                  # SHARED driver: python optimize.py --config-dir <folder> --device <x>
├── cpu/        (fp16)   embedding.json  text.json  vision.json  eval.py  inference.py
├── webgpu/     (int4)   embedding.json  text.json  vision.json  eval.py  inference.py
├── cuda/       (fp32)   embedding.json  text.json  vision.json  eval.py  inference.py
└── openvino/   (fp32, UNVERIFIED EP — see §4)
                          embedding.json  text.json  vision.json  eval.py  inference.py
```

**Precision is applied UNIFORMLY per folder** (all 3 components at that folder's stated
precision) — e.g. `webgpu/` RTN-quantizes embedding AND vision to int4, not just the text
decoder. This is a deliberate simplification of the existing `Qwen-Qwen3.5-0.8B/builtin`
pattern, which keeps embedding/vision at fp16 even on its int4 webgpu target (only the LLM
gets int4 there) — flagging this divergence explicitly in case it should instead follow that
finer-grained precedent.

`eval.py`/`inference.py` are otherwise **identical copies** across all 4 folders — the only
per-folder edit was the `--model_path` default (`<folder>/models`). Both are genai-runtime
driven (`onnxruntime_genai.Model(model_path)` + the assembled `genai_config.json`), not raw
ONNX sessions — model-agnostic, no changes needed for 27B vs 0.8B beyond the model id.

## 4. Traps inherited from the qwen3_5 family (see `olive-model-recipes` skill §Cross-project
   traps for the general form of each)

- **Trap #2 (MROPE / ModelBuilder gaps)**: genai `create_model()` crashes with
  `KeyError('root_input')` in `make_packed_attention` on this hybrid's full-attention layers.
  Every `text.json` here uses the **Olive `ModelBuilder` pass**, not direct `create_model()` —
  confirmed this is what worked for Surya-2/chandra-ocr-2. Don't "simplify" this to
  `create_model()`.
- **Trap #6 (tokenizer regex)**: `fix_tokenizer()` in `optimize.py` (copied verbatim from the
  0.8B template) strips the `\p{L}`/`\p{N}` Unicode-property Split pre-tokenizer that C++
  `std::regex` in onnxruntime-genai can't parse, keeping only ByteLevel. Still needed here —
  Qwen3.8-27B's `tokenizer.json` should be checked/patched the same way once actually exported.
- **Trap #1 (RAM wall)**: at 27B, the int4 ModelBuilder serialize (webgpu target only, here —
  cpu/cuda/openvino are fp16/fp32) will need considerably more free RAM than chandra-8B's
  ~16-20 GB. Budget accordingly; this is a bigger model than anything in that table.
- **§4 openvino EP — genuinely unverified, not just "untested".** No existing recipe in this
  repo uses `OpenVINOExecutionProvider`. The `openvino/*.json` configs here are a best-effort
  guess (`device: "cpu"`, `execution_providers: ["OpenVINOExecutionProvider"]`) and
  `optimize.py`'s `update_genai_config` guesses `provider_options = [{"openvino": {}}]` for the
  genai session — **neither has been confirmed against an actual Olive/onnxruntime-genai run.**
  Check both against the installed Olive/genai versions' actual supported EP/provider-option
  schemas before trusting this target; it may need real correction, not just a first try.

## 5. Recon evidence trail

Weight census (`model.safetensors.index.json`, 1199 tensors, fully accounted for):
`language_model.embed_tokens`(1) + `.layers`(848, 64×13-14) + `.norm`(1) + `lm_head`(1) +
`visual.{patch_embed,blocks,merger,pos_embed}`(333) + `mtp.*`(14) = 1199. Layer 3 sampled and
confirmed `full_attention` (self_attn.{q,k,v,o}_proj + q_norm/k_norm); layers 0/1/2/4 confirmed
`linear_attention` (conv1d/in_proj_*/A_log/dt_bias) — matches `full_attention_interval: 4`
exactly.

## THE CHECKLIST — what's done vs pending

**Done (2026-09-28):**
- [x] Recon: config.json dump, weight-group census, actual-vs-configured layer count check.
- [x] 4 folders scaffolded with valid Olive JSON configs (all 12 parse-checked).
- [x] Shared `user_script.py`/`codes/modeling_qwen3_5.py`/`optimize.py` adapted from the
      Qwen3.5-0.8B precedent (model id, token ids, `out_hidden_size`, no `TieWordEmbeddings`).
- [x] Per-folder `eval.py`/`inference.py` (AI2D-benchmark-based, inherited from the 0.8B
      precedent — **not yet confirmed as the right benchmark for THIS model**; revisit if a
      different eval task is wanted).

**Pending — nothing below this line has been run:**
- [ ] Decide the MTP-head question for real (currently: skip). If wrong, `codes/modeling_qwen3_5.py`
      needs an `mtp` submodule added and a 4th export target.
- [ ] Actually run `python optimize.py --config-dir cpu --device cpu` (etc. per folder) — this
      is where the ~55 GB weight download actually happens, and where every guess above
      (openvino EP, precision-per-folder choice) gets tested against reality.
- [ ] Parity: none of the exported ONNX parts have been compared to PyTorch yet. Every
      cross-project trap doc insists on this before trusting a build — do not skip it here
      just because the JSON configs look right on paper.
- [ ] Confirm `fix_tokenizer`'s regex strip is actually needed for THIS tokenizer.json (verify,
      don't assume it transfers unchanged from 0.8B).
- [ ] Update this file + `STATUS.md` (not yet created — add one on first real build) with
      whatever new trap the first actual build run surfaces. That's the contract.
