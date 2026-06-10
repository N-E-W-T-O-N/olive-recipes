# INT4 on CPU (no GPU required)
python create.py --device cpu

# INT4 on CUDA GPU
python create.py --device gpu

# fp16 on GPU
python create.py --device gpu --precision fp16

# Use a local model folder instead of downloading
python create.py --device gpu --input D:/models/sarvam-30b

# Custom output path
python create.py --device gpu --output D:/output/sarvam-onnx

# Config/GenAI files only (skip ONNX conversion)
python create.py --device gpu --config_only

# Pass extra builder options
python create.py --device gpu --extra_options int4_algo_config=k_quant_mixed qmoe_block_size=128