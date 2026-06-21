# Sigmoid — Logistic 激活 (对应 02_intermediate/01_sigmoid.mlir)
# MLIR: negf + exp + addf + divf
# Triton: tl.sigmoid 内建 (1/x: tl.sigmoid)
# 公式: sigmoid(x) = 1 / (1 + e^{-x})

import triton
import triton.language as tl
import torch

@triton.jit
def sigmoid_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.sigmoid(x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
    y = torch.empty(5, device='cuda')
    sigmoid_kernel[(1,)](x, y, BLOCK=32)
    print(f"✅ Sigmoid: {dict(zip(x.cpu().tolist(), [f'{v:.4f}' for v in y.cpu().tolist()]))}")
