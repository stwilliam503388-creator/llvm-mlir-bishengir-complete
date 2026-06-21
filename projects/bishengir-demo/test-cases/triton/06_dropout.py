# Dropout — 随机丢弃 (对应 01_basic/06_dropout.mlir)
# MLIR: arith.mulf (简化版)
# Triton: tl.rand + tl.where
# 公式: y = x * scale (简化); 完整: mask * p 训练时

import triton
import triton.language as tl
import torch

@triton.jit
def dropout_kernel(X, Y, scale: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = x * scale
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda')
    y = torch.empty(128, device='cuda')
    dropout_kernel[(4,)](x, y, scale=1.25, BLOCK=32)
    print(f"✅ Dropout: mean(x)={x.mean().item():.4f}, mean(y)={(y/1.25).mean().item():.4f}")
