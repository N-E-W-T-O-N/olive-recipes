"""
eval.py — CSM-1B TTS Evaluation
================================
Runs both the original sesame/csm-1b model and the Olive-converted model,
evaluates on standard TTS test sentences, and produces a side-by-side
accuracy / quality comparison report.

Metrics
-------
  WER          Word Error Rate (Whisper ASR transcription vs. input text)
  CER          Character Error Rate
  UTMOS        Predicted MOS score (UTMOSv2)
  MCD          Mel Cepstral Distortion between original and converted outputs
  SpeakerSim   Cosine speaker-embedding similarity (SpeechBrain X-Vector)
  RTF          Real-Time Factor  (generation time / audio duration)

Install dependencies
--------------------
  pip install torch torchaudio transformers
  pip install openai-whisper jiwer
  pip install utmos
  pip install pymcd
  pip install speechbrain
  pip install onnxruntime   # or onnxruntime-gpu
  pip install tabulate

Usage
-----
  # Evaluate original model only
  python eval.py --device cpu

  # Evaluate original + converted ONNX model
  python eval.py --device cuda --onnx_dir model/cuda_fp16

  # Use fewer test sentences for a quick check
  python eval.py --device cpu --num_sentences 5

  # Save generated audio files
  python eval.py --device cuda --save_audio
"""

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Standard TTS test sentences
# (Harvard sentences + common TTS benchmarks covering diverse phonemes)
# ---------------------------------------------------------------------------
TEST_SENTENCES = [
    # Harvard sentences (IEEE recommended for speech quality testing)
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "Large size in stockings is hard to sell.",
    # Phoneme-diverse sentences
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "We promptly judged antique ivory buckles for the next prize.",
    # Short and long utterances
    "Hello.",
    "Good morning, how are you today?",
    "The advancement of artificial intelligence is reshaping industries worldwide.",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    sentence: str
    audio: np.ndarray          # float32, shape [T]
    sample_rate: int
    generation_time_s: float
    rtf: float
    wer: float = float("nan")
    cer: float = float("nan")
    utmos: float = float("nan")
    mcd: float = float("nan")           # vs reference (original model)
    speaker_sim: float = float("nan")   # vs reference


@dataclass
class EvalReport:
    model_tag: str
    results: list[TTSResult] = field(default_factory=list)

    def avg(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results
                if not np.isnan(getattr(r, attr))]
        return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Lazy imports (only load heavy libraries when needed)
# ---------------------------------------------------------------------------

def _import_whisper():
    try:
        import whisper
        return whisper
    except ImportError:
        print("[WARN] openai-whisper not installed. Skipping WER/CER.")
        return None

def _import_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError:
        print("[WARN] jiwer not installed. Skipping WER/CER.")
        return None

def _import_utmos():
    try:
        import utmos
        return utmos
    except ImportError:
        print("[WARN] utmos not installed. Skipping UTMOS.")
        return None

def _import_pymcd():
    try:
        from pymcd.mcd import Calculate_MCD
        return Calculate_MCD
    except ImportError:
        print("[WARN] pymcd not installed. Skipping MCD.")
        return None

def _import_speechbrain():
    try:
        from speechbrain.pretrained import SpeakerRecognition
        return SpeakerRecognition
    except ImportError:
        print("[WARN] speechbrain not installed. Skipping SpeakerSim.")
        return None


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

class OriginalCSMModel:
    """Wraps sesame/csm-1b via the HuggingFace transformers API."""

    MODEL_ID = "sesame/csm-1b"

    def __init__(self, device: str = "cpu"):
        print(f"[INFO] Loading original model: {self.MODEL_ID} on {device}")
        from transformers import CsmForConditionalGeneration, AutoProcessor
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = CsmForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.sample_rate = 24_000   # Mimi codec native rate

    def generate(self, text: str) -> tuple[np.ndarray, float]:
        """Returns (audio_f32_numpy, generation_time_s)."""
        conversation = [
            {
                "role": "0",
                "content": [{"type": "text", "text": text}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            audio_out = self.model.generate(**inputs, output_audio=True)
        gen_time = time.perf_counter() - t0

        # processor.save_audio returns tensor; convert to numpy
        audio_np = audio_out.squeeze().float().cpu().numpy()
        return audio_np, gen_time


class ONNXCSMModel:
    """Runs an Olive-converted ONNX model via onnxruntime."""

    def __init__(self, onnx_dir: str, device: str = "cpu"):
        import onnxruntime as ort
        from transformers import AutoProcessor

        self.sample_rate = 24_000
        self.device = device
        self.processor = AutoProcessor.from_pretrained("sesame/csm-1b")

        # Find the ONNX model file
        onnx_candidates = list(Path(onnx_dir).rglob("*.onnx"))
        if not onnx_candidates:
            raise FileNotFoundError(f"No .onnx files found under {onnx_dir}")
        onnx_path = str(onnx_candidates[0])
        print(f"[INFO] Loading ONNX model: {onnx_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device != "cpu"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def generate(self, text: str) -> tuple[np.ndarray, float]:
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": text}]}
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        )
        feed = {k: v.numpy() for k, v in inputs.items()}

        t0 = time.perf_counter()
        outputs = self.session.run(None, feed)
        gen_time = time.perf_counter() - t0

        audio_np = outputs[0].squeeze().astype(np.float32)
        return audio_np, gen_time


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, whisper_model: str = "base"):
        print("[INFO] Loading evaluation models...")
        self._whisper = None
        self._jiwer = None
        self._utmos = None
        self._mcd_fn = None
        self._spk_model = None
        self._whisper_size = whisper_model
        self._load_all()

    def _load_all(self):
        whisper = _import_whisper()
        if whisper:
            print(f"  Loading Whisper ({self._whisper_size})...")
            self._whisper = whisper.load_model(self._whisper_size)
        self._jiwer = _import_jiwer()

        utmos_lib = _import_utmos()
        if utmos_lib:
            print("  Loading UTMOS scorer...")
            self._utmos = utmos_lib.load_score_listener("strong", "cuda" if torch.cuda.is_available() else "cpu")

        self._mcd_fn = _import_pymcd()

        SpeakerRecognition = _import_speechbrain()
        if SpeakerRecognition:
            print("  Loading SpeechBrain X-Vector speaker model...")
            self._spk_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="cache/speechbrain",
            )

    # ------------------------------------------------------------------
    def wer_cer(self, reference: str, audio_np: np.ndarray, sr: int) -> tuple[float, float]:
        if not (self._whisper and self._jiwer):
            return float("nan"), float("nan")
        audio_16k = _resample(audio_np, sr, 16_000)
        result = self._whisper.transcribe(audio_16k, language="en")
        hyp = result["text"].strip()
        wer = self._jiwer.wer(reference.lower(), hyp.lower())
        cer = self._jiwer.cer(reference.lower(), hyp.lower())
        return wer, cer

    def utmos_score(self, audio_np: np.ndarray, sr: int) -> float:
        if not self._utmos:
            return float("nan")
        audio_16k = _resample(audio_np, sr, 16_000)
        tensor = torch.tensor(audio_16k).unsqueeze(0)
        score = self._utmos.score(tensor, 16_000)
        return float(score)

    def mcd(self, ref_np: np.ndarray, hyp_np: np.ndarray, sr: int, tmp_dir: str = "cache/tmp_mcd") -> float:
        if not self._mcd_fn:
            return float("nan")
        os.makedirs(tmp_dir, exist_ok=True)
        ref_path = os.path.join(tmp_dir, "ref.wav")
        hyp_path = os.path.join(tmp_dir, "hyp.wav")
        _save_wav(ref_np, sr, ref_path)
        _save_wav(hyp_np, sr, hyp_path)
        mcd_val, _ = self._mcd_fn(ref_path, hyp_path, "dtw")
        return float(mcd_val)

    def speaker_sim(self, ref_np: np.ndarray, hyp_np: np.ndarray, sr: int) -> float:
        if not self._spk_model:
            return float("nan")
        ref_t = torch.tensor(_resample(ref_np, sr, 16_000)).unsqueeze(0)
        hyp_t = torch.tensor(_resample(hyp_np, sr, 16_000)).unsqueeze(0)
        score, _ = self._spk_model.verify_batch(ref_t, hyp_t)
        return float(score.item())


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _resample(audio_np: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio_np
    t = torch.tensor(audio_np).unsqueeze(0)
    t = torchaudio.functional.resample(t, src_sr, tgt_sr)
    return t.squeeze().numpy()


def _save_wav(audio_np: np.ndarray, sr: int, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = torch.tensor(audio_np).unsqueeze(0)
    torchaudio.save(path, t, sr)


# ---------------------------------------------------------------------------
# Single-model evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    sentences: list[str],
    evaluator: Evaluator,
    model_tag: str,
    reference_results: Optional[list[TTSResult]] = None,
    save_audio: bool = False,
    audio_dir: str = "eval_audio",
) -> EvalReport:
    report = EvalReport(model_tag=model_tag)

    for i, sentence in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {sentence[:60]}...")
        audio_np, gen_time = model.generate(sentence)
        duration_s = len(audio_np) / model.sample_rate
        rtf = gen_time / duration_s if duration_s > 0 else float("nan")

        result = TTSResult(
            sentence=sentence,
            audio=audio_np,
            sample_rate=model.sample_rate,
            generation_time_s=gen_time,
            rtf=rtf,
        )

        # WER / CER
        result.wer, result.cer = evaluator.wer_cer(sentence, audio_np, model.sample_rate)

        # UTMOS
        result.utmos = evaluator.utmos_score(audio_np, model.sample_rate)

        # MCD + SpeakerSim vs reference (original model output)
        if reference_results and i < len(reference_results):
            ref = reference_results[i]
            result.mcd = evaluator.mcd(ref.audio, audio_np, model.sample_rate)
            result.speaker_sim = evaluator.speaker_sim(ref.audio, audio_np, model.sample_rate)

        report.results.append(result)

        if save_audio:
            tag = model_tag.replace(" ", "_").replace("/", "-")
            path = os.path.join(audio_dir, tag, f"sentence_{i+1:02d}.wav")
            _save_wav(audio_np, model.sample_rate, path)
            print(f"    Saved: {path}")

    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(reports: list[EvalReport]):
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    print("\n" + "=" * 72)
    print("  CSM-1B TTS EVALUATION REPORT")
    print("=" * 72)

    # Summary table
    headers = ["Model", "WER ↓", "CER ↓", "UTMOS ↑", "MCD ↓", "SpeakerSim ↑", "RTF ↓"]
    rows = []
    for rep in reports:
        rows.append([
            rep.model_tag,
            f"{rep.avg('wer'):.4f}" if not np.isnan(rep.avg('wer')) else "N/A",
            f"{rep.avg('cer'):.4f}" if not np.isnan(rep.avg('cer')) else "N/A",
            f"{rep.avg('utmos'):.3f}" if not np.isnan(rep.avg('utmos')) else "N/A",
            f"{rep.avg('mcd'):.3f}"  if not np.isnan(rep.avg('mcd')) else "N/A",
            f"{rep.avg('speaker_sim'):.4f}" if not np.isnan(rep.avg('speaker_sim')) else "N/A",
            f"{rep.avg('rtf'):.4f}" if not np.isnan(rep.avg('rtf')) else "N/A",
        ])

    if has_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("  " + " | ".join(headers))
        for row in rows:
            print("  " + " | ".join(str(c) for c in row))

    # Degradation vs original
    if len(reports) >= 2:
        orig = reports[0]
        print("\n  Degradation vs Original (converted - original):")
        for rep in reports[1:]:
            wer_diff  = rep.avg("wer")  - orig.avg("wer")
            cer_diff  = rep.avg("cer")  - orig.avg("cer")
            mos_diff  = rep.avg("utmos") - orig.avg("utmos")
            rtf_diff  = rep.avg("rtf")  - orig.avg("rtf")
            mcd_val   = rep.avg("mcd")
            spk_val   = rep.avg("speaker_sim")

            print(f"\n  [{rep.model_tag}]")
            _print_delta("  WER",        wer_diff,  lower_is_better=True)
            _print_delta("  CER",        cer_diff,  lower_is_better=True)
            _print_delta("  UTMOS",      mos_diff,  lower_is_better=False)
            _print_delta("  RTF",        rtf_diff,  lower_is_better=True)
            if not np.isnan(mcd_val):
                print(f"  MCD (absolute):      {mcd_val:.3f}  (< 5 dB = near-identical)")
            if not np.isnan(spk_val):
                print(f"  SpeakerSim:          {spk_val:.4f}  (> 0.85 = high similarity)")

    # Per-sentence detail
    print("\n" + "-" * 72)
    print("  Per-sentence breakdown")
    print("-" * 72)
    for rep in reports:
        print(f"\n  Model: {rep.model_tag}")
        for i, r in enumerate(rep.results):
            label = r.sentence[:55] + "..." if len(r.sentence) > 55 else r.sentence
            wer_s = f"WER={r.wer:.3f}" if not np.isnan(r.wer) else "WER=N/A"
            mos_s = f"UTMOS={r.utmos:.2f}" if not np.isnan(r.utmos) else "UTMOS=N/A"
            rtf_s = f"RTF={r.rtf:.3f}"
            mcd_s = f"MCD={r.mcd:.2f}" if not np.isnan(r.mcd) else ""
            print(f"  {i+1:2d}. {label:<57}  {wer_s}  {mos_s}  {rtf_s}  {mcd_s}")

    print("\n" + "=" * 72)
    print("  Legend:")
    print("  WER  = Word Error Rate (0 = perfect transcription)")
    print("  CER  = Character Error Rate")
    print("  UTMOS= Predicted MOS 1-5 (5 = best quality)")
    print("  MCD  = Mel Cepstral Distortion in dB (lower = closer to reference)")
    print("  SpeakerSim = Cosine speaker similarity vs original (1.0 = identical)")
    print("  RTF  = Real-Time Factor (<1 = faster than real-time)")
    print("=" * 72 + "\n")


def _print_delta(label: str, delta: float, lower_is_better: bool):
    if np.isnan(delta):
        print(f"  {label:<20} N/A")
        return
    sign = "+" if delta > 0 else ""
    good = (delta <= 0) if lower_is_better else (delta >= 0)
    flag = "[OK]" if good else "[DEGRADED]"
    print(f"  {label:<20} {sign}{delta:+.4f}  {flag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CSM-1B original vs Olive-converted model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run models on (default: cpu)")
    parser.add_argument("--onnx_dir", default=None,
                        help="Path to Olive output directory containing .onnx file. "
                             "If omitted, only the original HF model is evaluated.")
    parser.add_argument("--num_sentences", type=int, default=len(TEST_SENTENCES),
                        help=f"Number of test sentences to use (default: {len(TEST_SENTENCES)})")
    parser.add_argument("--whisper_model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for WER (default: base)")
    parser.add_argument("--save_audio", action="store_true",
                        help="Save generated audio .wav files to eval_audio/")
    parser.add_argument("--audio_dir", default="eval_audio",
                        help="Directory to save audio files (default: eval_audio)")
    parser.add_argument("--output_report", default=None,
                        help="Save report to a text file in addition to printing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    sentences = TEST_SENTENCES[: args.num_sentences]

    print("=" * 72)
    print("  CSM-1B TTS Evaluation")
    print("=" * 72)
    print(f"  Device          : {args.device}")
    print(f"  Test sentences  : {len(sentences)}")
    print(f"  Whisper model   : {args.whisper_model}")
    print(f"  ONNX dir        : {args.onnx_dir or '(not provided — original only)'}")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Load evaluation tools
    # ------------------------------------------------------------------
    evaluator = Evaluator(whisper_model=args.whisper_model)

    # ------------------------------------------------------------------
    # Evaluate original model
    # ------------------------------------------------------------------
    print("\n[1/2] Evaluating original model (sesame/csm-1b)...")
    orig_model = OriginalCSMModel(device=args.device)
    orig_report = evaluate_model(
        orig_model, sentences, evaluator,
        model_tag="sesame/csm-1b (original)",
        reference_results=None,
        save_audio=args.save_audio,
        audio_dir=args.audio_dir,
    )

    reports = [orig_report]

    # ------------------------------------------------------------------
    # Evaluate converted ONNX model (if provided)
    # ------------------------------------------------------------------
    if args.onnx_dir:
        print(f"\n[2/2] Evaluating converted ONNX model ({args.onnx_dir})...")
        try:
            onnx_model = ONNXCSMModel(onnx_dir=args.onnx_dir, device=args.device)
            onnx_report = evaluate_model(
                onnx_model, sentences, evaluator,
                model_tag=f"ONNX ({Path(args.onnx_dir).name})",
                reference_results=orig_report.results,   # compute MCD vs original
                save_audio=args.save_audio,
                audio_dir=args.audio_dir,
            )
            reports.append(onnx_report)
        except Exception as e:
            print(f"[ERROR] Failed to load/run ONNX model: {e}")
            print("        Continuing with original model results only.")

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    if args.output_report:
        import io
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        print_report(reports)
        sys.stdout = old_stdout
        text = buf.getvalue()
        print(text)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[INFO] Report saved to: {args.output_report}")
    else:
        print_report(reports)


if __name__ == "__main__":
    main()
