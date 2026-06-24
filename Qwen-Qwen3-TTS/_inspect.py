# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==4.57.3","torch","olive-ai","onnx","onnxruntime>=1.20","numpy"]
# ///
import inspect, re
import olive.passes.onnx.conversion as C
print(inspect.getsource(C._convert_dynamic_shapes_for_dynamic_cache))
src = inspect.getsource(C)
# find past_key_values config param + how dummy/io names handled
for m in re.finditer(r'(past_key_value\w*|kv_cache_dtype|past_names)[^\n]*', src):
    print("CFG:", m.group(0)[:120])
