1|# (对应 mlir/02_intermediate/11_layer_norm.mlir) ⭐⭐
2|# 公式: y = (x - mu) / sqrt(var + eps) * gamma + beta
3|# 一句话: 每个 token 自己标准化, 不依赖 batch
4|# 专业角色: Transformer 标配归一化, 适合变长序列
5|# 用在哪: GPT/BERT/LLaMA 每层 Attention 和 FFN 后
6|# 降级对比: MLIR subf+mulf, Triton tl.mean+tl.var (5行)
7|# bishengir: 组合 (配合 reduce+broadcast)
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def layer_norm_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    mask = offsets < N
17|    x = tl.load(X + offsets, mask=mask)
18|    mean = tl.sum(x, axis=0) / N
19|    diff = x - mean
20|    var = tl.sum(diff * diff, axis=0) / N
21|    y = diff / tl.sqrt(var + 1e-5)
22|    tl.store(Y + offsets, y, mask=mask)
23|
24|if __name__ == "__main__":
25|    x = torch.randn(128, device='cuda') * 2 + 1  # 非标准分布
26|    y = torch.empty(128, device='cuda')
27|    layer_norm_kernel[(1,)](x, y, N=128, BLOCK=128)
28|    print(f"✅ LayerNorm: mean={y.mean().item():.4f}, std={y.std(unbiased=False).item():.4f}")