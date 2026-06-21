# Clamp — 数值裁剪 (对应 02_intermediate/10_clamp.mlir)
# MLIR: cmpf + select x2
# Triton: tl.clamp 内建
# 公式: y = clamp(x, -1, 1)

import triton
import triton.language as tl
import torch

@triton.jit
def clamp_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.clamp(x, -1.0, 1.0)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.5, 0.0, 0.5, 2.0], device='cuda')
    y = torch.empty(5, device='cuda')
    clamp_kernel[(1,)](x, y, BLOCK=5)
    expected = torch.clamp(x, -1, 1)
    torch.testing.assert_close(y, expected)
    print(f"✅ Clamp: {y.tolist()}")
