# Fill — 张量填充 (对应 01_basic/07_fill.mlir)
# MLIR: linalg.generic + yield 常数
# Triton: 直接将常量存储到每个位置
# 公式: A[i][j] = c

import triton
import triton.language as tl
import torch

@triton.jit
def fill_kernel(X, val: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offsets, val)

if __name__ == "__main__":
    x = torch.empty(32, device='cuda')
    fill_kernel[(1,)](x, val=0.0, BLOCK=32)
    assert (x == 0).all(), f"Not all zeros: {x}"
    print(f"✅ Fill: 全部为 {x[0].item()}, sum={x.sum().item()}")
