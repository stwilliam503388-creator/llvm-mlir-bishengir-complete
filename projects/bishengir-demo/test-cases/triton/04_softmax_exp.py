# Softmax (exp) — 指数部分 (对应 01_basic/04_softmax_exp.mlir)
# MLIR: math.exp
# Triton: tl.exp
# 公式: y = e^{x}

import triton
import triton.language as tl
import torch

@triton.jit
def exp_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.exp(x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.empty(3, device='cuda')
    exp_kernel[(1,)](x, y, BLOCK=32)
    torch.testing.assert_close(y, torch.exp(x))
    print(f"✅ Softmax(exp): {y.tolist()}")
