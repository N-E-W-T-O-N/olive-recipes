# /// script
# requires-python = ">=3.10"
# dependencies = ["torch","safetensors","numpy","onnx","onnxscript"]
# ///
"""Bake the Qwen3 text-embedding table into a standalone ONNX (input_ids -> inputs_embeds).

This removes the runtime dependency on the original PyTorch model / qwen3_standalone:
the llm_decoder was exported with exclude_embeds, so the embedding lookup (a Gather)
must live somewhere -- here it becomes its own ONNX, shipped in the model dir.
"""
import sys
from pathlib import Path
import torch, torch.nn as nn
from safetensors import safe_open

HERE = Path(__file__).parent
SRC = HERE / "model" / "model.safetensors"
KEY = "tied.embedding.text_embedding.weight"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "cpu" / "cpu_int4")
OUT = OUT_DIR / "text_embed.onnx"


class Emb(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.emb = nn.Embedding(*w.shape)
        self.emb.weight.data = w

    def forward(self, input_ids):
        return self.emb(input_ids)


def main():
    with safe_open(str(SRC), framework="pt") as f:
        w = f.get_tensor(KEY).float()
    V, D = w.shape
    print(f"  embedding {V}x{D}")
    m = Emb(w).half().eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m, (ids,), str(OUT),
        input_names=["input_ids"], output_names=["inputs_embeds"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "inputs_embeds": {0: "batch", 1: "seq"}},
        opset_version=20)
    mb = OUT.stat().st_size / 1e6
    print(f"  wrote {OUT}  ({mb:.0f} MB, fp16)")


if __name__ == "__main__":
    main()
