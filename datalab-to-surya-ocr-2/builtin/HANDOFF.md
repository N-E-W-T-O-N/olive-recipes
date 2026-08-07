# surya-ocr-2 → ONNX — Engineer's Handoff Note

> **Path note (2026-07-30):** this project now lives at `Surya-2/` directly — the earlier
> `Surya-2/builtin/` nesting was flattened to match the original bare `Surya-2/` reference.
> `Surya-2/builtin/model/` is a stray leftover (duplicate model download, harmless, unused).

## 1. Mental model

**datalab-to/surya-ocr-2** is an OCR VLM built on **Qwen3.5-VL** (`model_type: qwen3_5`,
`Qwen3_5ForConditionalGeneration`). It is the **same architecture family as
`Qwen-Qwen3.5-0.8B/builtin/`** — this recipe started as a port of that one, then was refactored
(2026-07-29) to the **Python-driven `chandra/builtin`-style** pipeline (no static per-target JSON;
see §3). It is **not** the old `chandra/builtin` (Qwen3-VL-8B, stock full attention, built via
genai `create_model` directly) — do not reuse that project's `build_text` path here.

Text decoder = **hybrid**: 24 layers, `layer_types` = 3×`linear_attention` (GatedDeltaNet) + 1×
`full_attention`, repeating (`full_attention_interval: 4`). `attn_output_gate: true`,
`head_dim: 256`, 8 q-heads / 2 kv-heads, `partial_rotary_factor: 0.25`, MROPE interleaved
`mrope_section: [11,11,10]`. Dims (hidden 1024, vision depth 12, out_hidden 1024) are **identical
to the 0.8B base** — only the tokenizer and special-token ids differ.

Three ONNX sub-parts, re-composed by onnxruntime-genai at runtime via `genai_config.json`:
1. **text** — the hybrid decoder, built by the **Olive ModelBuilder pass** (`_text_config` in
   `optimize.py`), int4 `k_quant_linear` (or fp16/fp32), `prune_lm_head` + `TieWordEmbeddings`
   surgery.
2. **vision** — Olive dynamo export + graph surgeries (PackedAttentionToLoopMHA,
   ReciprocalMulToDiv on gpu, RenameOutputDims) + precision pass.
3. **embedding** — Olive legacy export (+GemmToMatMulAdd) + precision pass; fuses text embeds +
   image features.

Custom export-friendly model code: `codes/modeling_qwen3_5.py` (config-driven; shared verbatim
with the 0.8B and chandra-ocr-2 recipes).

## 2. What differs from the 0.8B recipe architecturally

1. **Model id** — `datalab-to/surya-ocr-2` (`MODEL_ID` in `optimize.py`; local `./model` preferred,
   see §3).
2. **Special-token ids** (`optimize.py::update_genai_config`) — surya ships a CUSTOM
   **65425-vocab, character-level tokenizer** with special tokens remapped to LOW ids (from
   `tokenizer.json` added_tokens):

   | token | id |
   |---|---|
   | `<|endoftext|>` (pad) | 0 |
   | `<|im_start|>` | 1 |
   | `<|im_end|>` (eos) | 2 |
   | `<|vision_start|>` | 9 |
   | `<|vision_end|>` | 10 |
   | `<|image_pad|>` (image_token_id) | 11 |
   | `<|video_pad|>` (video_token_id) | 12 |

   So: `eos=[2]`, `pad=0`, `image_token_id=11`, `video_token_id=12`, `vision_start_token_id=9`.
   **Config trap:** `config.json` `text_config.eos_token_id = 248044` is OUT OF VOCAB (hence the HF
   "eos must be within vocab" warning in the build log) — ignore it; the real values come from
   `generation_config.json` (eos=2, pad=0).
3. **`bos_token_id` MUST be a number, not null.** The model genuinely has no BOS
   (`config.json`'s `bos_token_id` is `null`), but genai's C++ `genai_config.json` parser
   **rejects a JSON `null` here** (`Expected a number but saw a null`) — confirmed by direct
   runtime error. Fix (mirrors the 0.8B recipe's own convention): set `bos_token_id = eos = 2` as
   an unused placeholder; it's never sampled since generation starts from the chat-template
   prompt, not from BOS.
4. **Tokenizer needed a real fix, not a no-op — see §5.4.** surya's `tokenizer.json` `model.type`
   is `WordLevel` (character-level, single `Split(Regex="." Isolated)` pre-tokenizer). genai's C++
   tokenizer runtime does NOT understand `WordLevel` — `fix_tokenizer` rewrites it to
   `type: "BPE"` with an **empty `merges` list**, which is functionally identical here (verified by
   real inference, §4).
5. `out_hidden_size` in `user_script.get_embedding_dummy_inputs` reads from config (== 1024).
   Vision dummy unchanged (patch dim 3·2·16·16 = 1536; depth 12 same as 0.8B).

`processor_config` (name `qwen2_5_image_processor`, mean/std 0.5, patch 16, merge 2, temporal 2)
matches surya's `Qwen2VLImageProcessorFast`. Resize is left at 960×672 smart-resize —
**tune for OCR page resolution** (documents want more pixels; larger images already push input
token counts past 6000+, see §4).

## 3. Recipe shape — Python-driven, not static JSON (2026-07-29 refactor)

Per user preference, `optimize.py` generates every Olive config **in Python** (dicts + a temp-file
`_run_olive` helper) instead of shipping static `cpu_and_mobile/`, `cuda/`, `webgpu/` JSON dirs (the
0.8B recipe's shape — those dirs were deleted here). Mirrors `chandra/builtin/optimize.py`'s
`--device` × `--precision` structure, but builds text via the **Olive ModelBuilder pass** (matching
the 0.8B sibling) rather than chandra's own bare `create_model()` call — a stylistic/consistency
choice, not a functional necessity: both call the identical underlying genai function, and neither
one avoids the real `cuda_fp32` limitation (see §5.1 for the precise, traced mechanism — it's about
`(execution_provider, io_dtype)`, not which Python entry point is used).

```bash
python optimize.py --device cpu   --precision int4   # all three, cpu — BUILT + TESTED
python optimize.py --device cpu   --precision fp32   # cpu, no int4 quant — BUILT + TESTED
python optimize.py --device cuda  --precision fp16   # gpu, fp16 (recommended for teamspace)
python optimize.py --device cuda  --precision int4
python optimize.py --device webgpu --precision fp16
python optimize.py --device cpu --precision int4 --components vision   # subset
python optimize.py --device cpu --precision int4 --skip-export         # regen configs only
python optimize.py --device cpu --precision int4 --no-isolate          # one process (default: isolated)
```

Valid `(device, precision)`: `cpu×{int4,fp32}`, `cuda×{int4,fp16}`, `webgpu×{int4,fp16}` —
**fp16 is GPU-only**; Olive's ModelBuilder pass rejects it on CPU (`FP16 is not supported on CPU`,
confirmed by probe on this box). **`cuda_fp32` is deliberately excluded** — architecturally broken
for this model, not just untested; see §5.1 for the exact traced mechanism. Output →
`<device>_<precision>/models/`. Model source: local
`./model` snapshot if present (what you download to), else fetches `MODEL_ID` from the Hub.
Component builds run in isolated subprocesses by default (fresh memory per component); pass
`--no-isolate` to run in one process.

## 4. Status — CPU int4 AND fp32 built + tested end-to-end with real inference (2026-07-29)

Both `cpu_int4` and `cpu_fp32` targets built via `optimize.py` (current stack: `transformers
5.14.1`, `onnxruntime-genai 0.14.1`, `torch 2.12.1+cpu`, `olive 0.13.0`), from the local `./model`
snapshot, subprocess-isolated (text → embedding → vision). int4: text.onnx 665 MB (~230s int4
serialize) + TieWordEmbeddings surgery, embedding.onnx 35 MB, vision.onnx 58 MB. fp32: same shape,
uncompressed. **No `KeyError`, no `root_input`** — direct regression test for the original blocker,
passed.

**Ran real inference (`inference.py`) against real document images shipped in `model/assets/`** —
this is the actual runtime test, not just an export check, and it exercises the multimodal
processor + tokenizer + KV-cache decode loop end to end:

| target | image | result |
|---|---|---|
| cpu_int4 | corporate_text.png | ✅ coherent layout JSON (Section-Header/Table-Of-Contents/Text, plausible bboxes) — 79 tokens, 30.5 tok/s |
| cpu_int4 | handwritten.png | ✅ coherent layout JSON (26 regions: Text/Table/Section-Header) — 415 tokens, 35.7 tok/s |
| cpu_int4 | form_text.png | ✅ coherent layout JSON (List-Group/Table/Section-Header) — needed `--max_length 8192` (6946 input tokens: larger image → more vision-patch tokens) |
| cpu_fp32 | corporate_text.png | ✅ near-identical structure/bboxes to the int4 run on the same image — cross-validates int4 didn't meaningfully degrade output |

Two bugs found and fixed by this testing (both now baked into `optimize.py`, not manual patches):

1. **`bos_token_id: null` crashes genai's JSON parser** — see §2.3. Fixed: `bos_token_id = 2`
   (= eos), matching the 0.8B recipe's own convention for BOS-less Qwen3.5 models.
2. **WordLevel tokenizer crashes genai's C++ tokenizer** at `create_multimodal_processor()`
   (`RuntimeError: Cannot know how to parse line: `) — confirmed via binary inspection of
   `onnxruntime-genai.dll` that its tokenizer only understands BPE-shaped `model` sections
   (expects a `merges` list; no `WordLevel`/`WordPiece`/`Unigram` string exists anywhere in the
   DLL). Fixed in `fix_tokenizer`: rewrite `model.type` to `"BPE"` with `merges: []`. This is
   **not a hack** — since the pre-tokenizer already isolates exactly one Unicode character per
   piece (`Split(Regex="." Isolated)`), BPE with zero merge rules degenerates to a direct
   single-character vocab lookup, identical to what WordLevel did. Confirmed correct by the
   generation results above (real, sensible OCR/layout output, not garbage).

`--max_length` needs headroom for the *input* too, not just generation — vision-patch tokens for a
single page can run 2500–7000+ depending on resolution; the runtime error when too small
(`input_ids size (N) + current sequence length (0) exceeds max length`) is self-explanatory, not a
bug.

## 5. Remaining open risks / traps

1. **genai text-decoder build — RESOLVED, root cause precisely traced (2026-07-30, corrects an
   earlier imprecise account below).** The original `KeyError('root_input')` was first attributed
   to "chandra's `create_model()` DIRECT path vs Olive's ModelBuilder pass" — that framing is
   **wrong**. Both call the exact same `onnxruntime_genai.models.builder.create_model()` (visible
   in Olive's own `model_builder.py::_run_for_config`, which imports and calls it directly). Traced
   through genai's actual builder source (`onnxruntime_genai/models/builders/base.py` +
   `qwen.py`) to the real mechanism:
   - `base.py::make_attention_init()` picks the attention op via `(execution_provider, io_dtype)`
     lookup tables: `is_gqa_supported()`'s allow-list has `("cuda", FLOAT16)`/`("cuda", BFLOAT16)`
     but **not** `("cuda", FLOAT)` (plain fp32) — that falls through to
     `is_packed_attn_supported()`, which *does* allow `("cuda", FLOAT)` → `op_type = "Attention"`
     (the packed op).
   - `qwen.py::_make_full_attention()` (Qwen3.5's doubled Q-projection/gating override, needed for
     this hybrid architecture) only ever calls
     `make_attention_op(..., q_path=, k_path=, v_path=...)` — it never supplies `root_input`. Fine
     for `GroupQueryAttention`/`MultiHeadAttention` (which read `q_path`/`k_path`/`v_path`), but
     `make_packed_attention()` unconditionally does `kwargs["root_input"]` → the KeyError, for
     this exact `(cuda, fp32)` combination specifically.
   - **This means `cuda_fp32` is architecturally broken for this model, regardless of Olive-pass
     vs direct `create_model()`** — removed from `VALID` in `optimize.py` (§3). `cuda_int4` and
     `cuda_fp16` both resolve `io_dtype=FLOAT16` (see `set_io_dtype`), which *is* GQA-safe, so
     they're unaffected. This is a gap in genai's own `qwen.py` Qwen3.5 override — not something
     this recipe can route around via config.
   - Every build in this project used `cpu` (`int4`/`fp32`, both resolve to `("cpu", FLOAT)`,
     which *is* GQA-safe) — that's the real reason they all worked, not "which Python entry point
     was used."
2. **WINDOWS build trap — set `PYTHONUTF8=1`.** The vision dynamo export (torch.onnx) prints a
   `✅` success line; on a cp1252 Windows stream that raises `UnicodeEncodeError: '✅'` and
   aborts the vision pass (text+embedding still complete). Not an issue on Linux/teamspace.
3. **CUDA build — cannot even be attempted on this dev box, AND `cuda_fp32` must not be attempted
   anywhere (see #1).** This box's `onnxruntime` has only
   `['AzureExecutionProvider', 'CPUExecutionProvider']` compiled in — no CUDA EP at all, independent
   of GPU hardware/drivers (confirmed via `onnxruntime.get_available_providers()` and
   `og.is_cuda_available() / is_dml_available()`, both `False`). Any `--device cuda` run here fails
   immediately in Olive's GPU-targeted passes. Must build AND test on the teamspace GPU box:
   `python optimize.py --device cuda --precision fp16` (or `int4` — **not** `fp32`, see #1). Expect
   it to work (same genai, same tokenizer fix); watch the CUDA-graph session_options split in
   `update_genai_config`.
4. **WordLevel tokenizer at runtime — RESOLVED, including the `ocr`-task edge case.** See §4 and
   §7.4 (root-caused: id 10147 = U+FFFD, missing from genai's reverse vocab map; fixed with a
   Python-side fallback decoder).
5. **`eval.py` was fully rewritten (2026-07-30)** — see §7.5. The old AI2D multiple-choice harness
   is gone.

## 6. Checklist

- [x] CPU int4 build (Python-driven) produces all three ONNX parts + patched genai_config. ✓
- [x] CPU fp32 build — same. ✓
- [x] Runtime smoke test on real document images (3 images, int4 + fp32) — coherent structured
      output, no errors once §4's two bugs were fixed. ✓
- [x] Real task prompts sourced from datalab-to/surya's actual code, `eval.py` rewritten around
      them (§7). ✓
- [x] `ocr`-task tokenizer crash root-caused (id 10147 = U+FFFD) and fixed with a fallback decoder
      (§7.4); verified on the exact real sequence that crashed 100% of the time. ✓
- [ ] CUDA build on teamspace: `python optimize.py --device cuda --precision fp16` (or `int4`) —
      cannot be attempted on this box at all (§5.3).

## 7. 2026-07-30 session — real prompts, precision-matrix limits, eval.py rewrite

### 7.1 int8 is not achievable for this text decoder

User asked for a cpu×cuda×{int4,int8,fp16,fp32} build matrix. Checked genai's ModelBuilder source
directly (`onnxruntime_genai/models/builder.py`): `argparse` `--precision` `choices=["int4", "bf16",
"fp16", "fp32"]` — **int8 is not an option at all**, for any execution provider. Per user decision,
int8 is dropped from the matrix entirely (not faked via a different quant path for just
vision/embedding). Matrix stays: `cpu×{int4,fp32}`, `cuda×{int4,fp16,fp32}`, `webgpu×{int4,fp16}`.

### 7.2 The real task prompts (sourced from datalab-to/surya's actual GitHub code)

My earlier smoke test used a paraphrased prompt ("Read the text in this image.") and the WRONG
input images (the README's `*_layout/_text/_reading/_tablerec.png` files are reference-OUTPUT
visualizations, not inputs — the real inputs are the unsuffixed pages). It still produced
plausible-looking layout JSON, which is exactly the trap: looked right, wasn't. Fetched
`surya/inference/prompts.py` from github.com/datalab-to/surya directly; its own comment says "the
exact wording is the model's training-time contract — do not paraphrase without retraining." Now
lives verbatim in **`tasks.py`** (new file, shared by `inference.py` and `eval.py`):

| task | prompt | max_tokens |
|---|---|---|
| `layout` | `Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.` | 3072 |
| `ocr` (surya's "high_accuracy_bbox", full page) | `OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).` | 12288 |
| `table_rec` | `Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).` | 3072 |
| `block` (per-cropped-region OCR — needs a prior layout pass, not used by eval.py) | `OCR this block image to HTML.` | 8192 |

Real input pages, from `model/assets/` (unsuffixed): `corporate.png`, `excerpt.png`, `form.png`,
`handwritten.png`, `newspaper.png`, `textbook.png`. **`table_rec` should only be tested on
`corporate.png` / `form.png` / `handwritten.png`** — surya's own README Examples table has no
Table-Rec column for excerpt/newspaper/textbook, meaning those pages have no table to recognize;
running table_rec on them isn't a meaningful test (confirmed: `excerpt.png` under `table_rec`
produced a `Page-Header` label, which isn't in `{"Row","Col"}` — the model correctly had nothing
table-like to report, not a bug). `eval.py`'s `TASK_IMAGE_NAMES` encodes this restriction.

`inference.py` now has `--task {layout,ocr,table_rec,block}` (uses the real prompt + its budgeted
max_length); `--prompt` still overrides it for ad-hoc use.

### 7.3 `qwen_vl_utils` isn't installed — and isn't needed

The inherited PyTorch-comparison code (`inference.py`'s `_run_pytorch`, old `eval.py`) imported
`qwen_vl_utils.process_vision_info`, which isn't in this venv (`ModuleNotFoundError`). Verified a
plain PIL image passed straight to `AutoProcessor.__call__(text=[...], images=[pil_image])` works
correctly (checked output shapes: `pixel_values [N,1536]`, `image_grid_thw`, sane `input_ids`) — no
video/multi-image handling needed for single-page eval. Removed the dependency; don't reintroduce
it without a concrete need (multi-image/video batching).

### 7.4 genai C++ tokenizer crash on `ocr` task — ROOT-CAUSED AND FIXED

`eval.py --task all` hit `RuntimeError: invalid map<K, T> key` during `ocr`/full-page generation
on `corporate.png`, identically on both `cpu_int4` and `cpu_fp32`. First investigation pass
(wrongly) concluded this was rare/nondeterministic CPU floating-point behavior, since a couple of
quick isolated repro attempts didn't reproduce it. **That conclusion was wrong** — a closer
investigation (six diagnostic scripts, each testing one hypothesis) found the true, 100%
deterministic cause:

1. Ruled out session/model reuse across prior generations (replaying the exact same 4-generation
   warm-up sequence before `ocr` still succeeded standalone) — NOT the cause.
2. Ruled out "just the bulk `tokenizer.decode(tokens)` call" — a version using only incremental
   `tokenizer_stream.decode()` per token *also* crashed, just at a different point (`del
   generator`'s C++ teardown). The crash point moves depending on exactly which decode call
   happens to be the *first* one to touch the bad token — not tied to one call site.
3. **Pinpointed the exact trigger**: wrapping the per-token `tokenizer_stream.decode()` call to
   report the offending id on failure found **token id 10147**, and confirmed even
   `tokenizer.decode([10147])` fails in total isolation — a fully deterministic, minimal repro.
4. **Identified what id 10147 actually is**: its vocab entry is the literal Unicode replacement
   character `U+FFFD` (`�`) — and critically, this entry is present in the **original, unmodified**
   `model/tokenizer.json` shipped by datalab, not something introduced by this recipe's
   WordLevel→BPE conversion (§2.4). genai's C++ tokenizer fails to index this one vocab entry into
   its internal reverse (id→string) map, so *any* attempt to decode a generated token with id
   10147 raises, regardless of which of genai's decode APIs is used or when.
5. Why only `ocr`: it's the only task that generates enough long, free-form real text that the
   model can legitimately emit a "low-confidence/unrecognized glyph" placeholder. `layout` and
   `table_rec`'s compact, constrained JSON output never emits it.

**Fix** (`inference.py`): `load_id_to_token(onnx_dir)` reads the built `tokenizer.json` directly
and builds a plain Python `{id: str}` map (vocab + added_tokens, added_tokens taking precedence on
overlapping ids — same resolution order HF tokenizers use). `generate_response` accepts this map
as `id_to_token` and uses it as a fallback **exactly when, and only when,** genai's own decode call
raises — at every point the crash was observed (per-token stream decode, bulk decode, generator
teardown). This is not a guess-and-substitute hack: the fallback string for id 10147 IS the correct
`U+FFFD` character, read from the same tokenizer.json genai itself failed to fully index. Wired
through every call site (`inference.py`'s CLI paths, `eval.py`'s per-`onnx_dir` loop).

**Verified fixed**: re-ran the exact real sequence that crashed 100% of the time
(`eval.py --task ocr --limit 2 cpu_int4/models`) — both images now complete cleanly (774 and 3268
tokens), producing coherent real HTML (`<div data-bbox="..." data-label="Table-Of-Contents">
<table>...`).

### 7.5 `eval.py` — full rewrite (the AI2D harness is gone)

Old `eval.py` loaded `lmms-lab/ai2d` (diagram multiple-choice QA) — completely irrelevant to an
OCR model, inherited unmodified from the 0.8B template. Replaced entirely per user-specified CLI:

```bash
python eval.py --model model cpu_int4/models cpu_fp32/models
python eval.py --model model                                    # auto-discovers built targets
python eval.py --model model --task layout --limit 5 cpu_int4/models
python eval.py --model model --skip-pytorch cpu_int4/models cpu_fp32/models
```

- `onnx_dirs` positional (nargs='*'): explicit list, or auto-discovers every `*/models` dir under
  the project with a `genai_config.json` if omitted.
- PyTorch baseline computed **once** and reused across every `onnx_dir` (not recomputed per
  target) — the expensive part shouldn't repeat.
- `layout`/`table_rec` outputs are checked against surya's real JSON schema (label set + bbox
  regex `x0 y0 x1 y1`), independent of any baseline — a correctness signal that doesn't need
  PyTorch at all.
- All tasks get a text-similarity score (`difflib.SequenceMatcher`) vs the PyTorch baseline when
  available.
- `--limit` (default 2) bounds images-per-task — full-page `ocr` is slow on CPU (seen: ~30 tok/s
  int4, ~12 tok/s fp32; a few thousand output tokens per page).
