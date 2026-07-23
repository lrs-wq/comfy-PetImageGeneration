from __future__ import annotations

import torch


assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
device = torch.device("cuda:0")
left = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
right = torch.eye(64, device=device)
result = left @ right
torch.cuda.synchronize()

assert torch.equal(left, result)
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"checksum={result.sum().item():.0f}")
