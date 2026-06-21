# GELU (tanh近似) (对应 02_intermediate/04_gelu_tanh.mlir)
# MLIR: math.tanh + addf + mulf
# Triton: tl.tanh 近似: 0.5*x*(1+tanh(x))
# 公式: gelu(x) ~= 0.5*x*(1+tanh(x))  [BERT/GPT-3 时代标准]

import triton
import triton.language as tl
import torch

@triton.jit
def gelu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    gelu = 0.5 * x * (1.0 + tl.tanh(x))
    tl.store(Y + offsets, gelu)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda')
    y = torch.empty(128, device='cuda')
    gelu_kernel[(4,)](x, y, BLOCK=32)
    expected = 0.5 * x * (1.0 + torch.tanh(x))
    torch.testing.assert_close(y, expected)
    print(f"✅ GELU: x[0]={x[0].item():.4f}, gelu={y[0].item():.4f}")
