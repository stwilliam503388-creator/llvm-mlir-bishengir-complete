# SiLU / Swish (对应 02_intermediate/02_silu.mlir)
# MLIR: sigmoid + mulf
# Triton: x * tl.sigmoid(x), LLaMA FFN 层标配
# 公式: silu(x) = x * sigmoid(x)

import triton
import triton.language as tl
import torch

@triton.jit
def silu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = x * tl.sigmoid(x)  # SiLU
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda')
    y = torch.empty(128, device='cuda')
    silu_kernel[(4,)](x, y, BLOCK=32)
    expected = x * torch.sigmoid(x)
    torch.testing.assert_close(y, expected)
    print(f"✅ SiLU: x[0]={x[0].item():.4f}, silu={y[0].item():.4f}")
