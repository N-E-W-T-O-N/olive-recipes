# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "numpy"]
# ///
"""Verify the *actual* numeric precision of exported ONNX sub-models.

File size is misleading: the codec parts (tok_encoder/tok_decoder) are forced
fp32 in every precision dir, so they're byte-identical across cpu_fp16 / cpu_fp32
/ cpu_int4. What really tells you the precision is the weight (initializer)
dtype histogram + presence of quant ops (MatMulNBits/DequantizeLinear for int4).

Usage:
  uv run check_precision.py onnx/cpu_int4 onnx/cpu_fp16 onnx/cpu_fp32
  uv run check_precision.py onnx/cpu_int4/talker.onnx        # single file
"""
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import numpy_helper, TensorProto

DT = {v: k for k, v in TensorProto.DataType.items()}
QUANT_OPS = {"MatMulNBits", "DequantizeLinear", "QuantizeLinear", "MatMulInteger",
             "DynamicQuantizeLinear", "ConvInteger"}


def inspect(path: Path):
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    # initializer dtype histogram, weighted by element count
    bytes_by_dt, count_by_dt = Counter(), Counter()
    for init in g.initializer:
        dt = DT.get(init.data_type, str(init.data_type))
        n = 1
        for d in init.dims:
            n *= d
        count_by_dt[dt] += n
        bytes_by_dt[dt] += 1
    ops = Counter(n.op_type for n in g.node)
    quant = {op: ops[op] for op in QUANT_OPS if op in ops}

    total = sum(count_by_dt.values()) or 1
    dt_summary = ", ".join(
        f"{dt}:{100*c/total:.1f}%" for dt, c in count_by_dt.most_common()
    )
    # infer label
    if quant:
        label = "INT4/quantized"
    elif count_by_dt.get("FLOAT16", 0) > count_by_dt.get("FLOAT", 0):
        label = "FP16"
    elif count_by_dt.get("FLOAT", 0) > 0:
        label = "FP32"
    else:
        label = "?"
    print(f"  {path.name:<22} -> {label}")
    print(f"      weight dtypes (by #elements): {dt_summary}")
    if quant:
        print(f"      quant ops: {quant}")


def main():
    targets = sys.argv[1:] or ["onnx/cpu_int4", "onnx/cpu_fp16", "onnx/cpu_fp32"]
    for t in targets:
        p = Path(t)
        files = sorted(p.glob("*.onnx")) if p.is_dir() else [p]
        print(f"\n=== {t} ===")
        for f in files:
            inspect(f)


if __name__ == "__main__":
    main()
