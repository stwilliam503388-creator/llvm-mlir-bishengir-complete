# Global Avg Pool (对应 03_advanced/07_global_avg_pool.mlir)
# MLIR: affine.for x2 + 累加 + 平均因子
# Triton: tl.sum / N
# 公式: 整个特征图求 1 个平均值

import triton
import triton.language as tl
import torch

@triton.jit
def gap_kernel(INPUT, OUTPUT, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(INPUT + offs)
    s = tl.sum(x) / N
    tl.store(OUTPUT, s)

if __name__ == "__main__":
    x = torch.randn(4, 4, device='cuda')
    out = torch.zeros(1, device='cuda')
    gap_kernel[(1,)](x.flatten(), out, N=16)
    expected = x.mean()
    print(f"✅ GAP: gap={out.item():.4f}, expected={expected.item():.4f}")
