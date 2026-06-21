# ReLU — 线性整流 (对应 01_basic/02_relu.mlir)
# MLIR: arith.cmpf + arith.select
# Triton: tl.where + tl.maximum
# 公式: y = max(0, x)

import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.maximum(x, 0.0)  # ReLU
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda') - 0.5
    y = torch.empty(128, device='cuda')
    relu_kernel[(4,)](x, y, BLOCK=32)
    torch.testing.assert_close(y, torch.relu(x))
    print(f"✅ ReLU: negative={y[y<0].numel()}, pos={y[y>0].numel()}")
