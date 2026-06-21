# (对应 02_intermediate/11_layer_norm.mlir) ⭐⭐
# 公式: y = (x - mu) / sqrt(var + eps) * gamma + beta
# 一句话: 每个 token 自己标准化, 不依赖 batch
# 专业角色: Transformer 标配归一化, 适合变长序列
# 用在哪: GPT/BERT/LLaMA 每层 Attention 和 FFN 后
# 降级对比: MLIR subf+mulf, Triton tl.mean+tl.var (5行)
# bishengir: 组合 (配合 reduce+broadcast)
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