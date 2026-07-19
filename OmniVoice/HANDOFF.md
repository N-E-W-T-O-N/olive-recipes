# OmniVoice → ONNX — Engineer's Handoff Note

_Read this before touching the project. Companion: the repo-wide skill at
`.claude/skills/olive-model-recipes/` and this file's checklist at the bottom._

## 1. Mental model

**Prince-1/OmniVoice** = zero-shot TTS (600+ languages), **non-autoregressive**: a Qwen3-0.6B
backbone run in a **32-step iterative-unmasking loop** over 8 audio codebooks (vocab 1025 =
1024 codes + MASK), 24 kHz. Voice cloning uses the **Higgs Audio V2 tokenizer** (4 extra
sub-models). We decompose the backbone into three ONNX parts and re-compose in `inference.py`:

1. `audio_embeddings_encoder` — text + codec token embeds → `inputs_embeds [B,S,1024]`
2. `llm_decoder` — Qwen3 core, `inputs_embeds → hidden_states` (no lm_head/embeds)
3. `audio_heads_decoder` — hidden → per-codebook logits `[B,8,S,1025]`

Plus (optional, `--include-higgs`): higgs acoustic/semantic encoders, quantizer, decoder.

## 2. Commands

```bash
python optimize.py --device cpu          # INT4 → cpu_and_mobile/
python optimize.py --device cpu_fp16     # FP16 → cpu_fp16/
python optimize.py --device gpu          # FP16 audio + INT4 LLM → cuda/
python optimize.py --include-higgs       # + 4 Higgs models → higgs/
python inference.py --text "Hello" --output out.wav --model_dir cpu_and_mobile/models
python inference.py --text "Say this" --ref_audio ref.wav --ref_text "..." --output cloned.wav
python eval.py --mode equiv --model_dir cpu_and_mobile/models --higgs_dir higgs/models
python eval.py --mode rtf   --model_dir cpu_and_mobile/models --compare cpu_fp16/models
```

## 3. Where things live

- `optimize.py` — `prepare_qwen3_standalone()` (HF dir for ModelBuilder),
  `export_llm_fp16_cpu()` (the FP16-on-CPU workaround), `export_models()` (Olive orchestration),
  `write_inference_manifest()` → `omnivoice_manifest.json` (exact I/O shapes + pipeline spec).
- `user_script.py` — Olive loaders; `save_qwen3_standalone()` patches config →
  `Qwen3ForCausalLM`; `_load_higgs_tokenizer()` + higgs wrappers; `_prepare_tok()`.
- `codes/model_wrappers.py` — the export wrappers + constants (SR 24k/16k, DOWNSAMPLE 960 → 25 fps).
- `inference.py` — `run_backbone_step()` (3-model forward), `iterative_unmask()` (32-step
  confidence-greedy loop), `higgs_encode()/higgs_decode()` (cloning).

## 4. Traps (hard-won)

1. **Olive ModelBuilder refuses FP16 on CPU EP.** Workaround: call onnxruntime-genai
   `create_model()` directly (`export_llm_fp16_cpu`, optimize.py ~lines 82–137).
2. **No KV reuse**: the 32-step loop feeds empty `past_key_values (B,heads,0,head_dim)` and does
   a full-sequence forward each step — that's by design for iterative unmasking, don't "optimize"
   it into an autoregressive cache without rethinking the algorithm.
3. **Higgs export needs `_prepare_tok()`** — strips weight_norm hooks and pre-traces the DAC
   encoder/decoder branches before ONNX export, else export fails.
4. Backbone uses the **legacy (non-dynamo) exporter, opset 20** in the Olive configs — keep it;
   the pipeline was validated on that path.
5. `omnivoice_manifest.json` is the source of truth for inference I/O — regenerate it whenever
   an export changes (`write_inference_manifest`).

## 5. Checklist — re-export / new target

- [ ] `python optimize.py --device <cpu|cpu_fp16|gpu>` (+ `--include-higgs` if cloning needed).
- [ ] `eval.py --mode equiv` → every sub-model numerically matches PyTorch.
- [ ] `eval.py --mode rtf` → record Real-Time Factor vs the previous build.
- [ ] Smoke `inference.py --text ...` (and a `--ref_audio` clone if higgs built) → listen.
- [ ] Manifest regenerated; no stale models dirs left behind.
