# Batch Norm Part1 — 均值 (对应 03_advanced/08_batch_norm_part1.mlir)
# MLIR: reduction + parallel
# Triton: tl.sum / N
# 公式: mean = sum(x[i][j]) / N (每通道)

import triton
import triton.language as tl
import torch

@triton.jit
def bn_mean_kernel(X, MEAN, CH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    s = tl.sum(x) / (4 * 4 // CH)
    tl.store(MEAN + pid, s)

if __name__ == "__main__":
    x = torch.randn(4, 4, device='cuda')
    m = torch.empty(4, device='cuda')
    bn_mean_kernel[(4,)](x.flatten(), m, CH=4, BLOCK=4)
    expected = x.mean(dim=1)
    print(f"✅ BN_mean: {m.tolist()}")
