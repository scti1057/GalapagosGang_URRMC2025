import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
	print("CUDA device name:", torch.cuda.get_device_name(0))
	print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")