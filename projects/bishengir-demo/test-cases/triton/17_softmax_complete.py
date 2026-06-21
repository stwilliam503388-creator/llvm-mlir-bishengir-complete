# Softmax 完整版 (对应 02_intermediate/09_softmax_complete.mlir)
# MLIR: reduce_max + exp(x-max)
# Triton: tl.exp(x - tl.max(x)) 一行完成
# 公式: e^{x - max(x)}  数值稳定的第一步

import triton
import triton.language as tl
import torch

@triton.jit
def softmax_stable_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    x_max = tl.max(x, axis=0)
    y = tl.exp(x - x_max)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([1000.0, 999.0, 998.0], device='cuda')
    y = torch.empty(3, device='cuda')
    softmax_stable_kernel[(1,)](x, y, BLOCK=32)
    print(f"✅ SoftmaxStable: {y.tolist()} (no overflow!)")
