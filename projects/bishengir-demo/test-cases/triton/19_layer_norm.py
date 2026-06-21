# Layer Norm 中间步 (对应 02_intermediate/11_layer_norm.mlir)
# MLIR: subf + mulf (diff^2)
# Triton: tl.var + tl.mean
# 公式: var = mean((x - mean(x))^2), 方差计算

import triton
import triton.language as tl
import torch

@triton.jit
def layer_norm_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    y = diff / tl.sqrt(var + 1e-5)
    tl.store(Y + offsets, y, mask=mask)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda') * 2 + 1  # 非标准分布
    y = torch.empty(128, device='cuda')
    layer_norm_kernel[(1,)](x, y, N=128, BLOCK=128)
    print(f"✅ LayerNorm: mean={y.mean().item():.4f}, std={y.std(unbiased=False).item():.4f}")
