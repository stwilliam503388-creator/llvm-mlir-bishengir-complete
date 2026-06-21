# (对应 03_advanced/09_batch_norm_part2.mlir) ⭐⭐⭐
# 公式: y = gamma*(x-mu)/sqrt(var+eps)+beta
# 一句话: 减去均值, 除以标准差, 缩放平移
# 专业角色: BN/LN/IN/GN 四种归一化通用第二步
# 用在哪: ResNet/CNN 全系 (归一化通用模式)
# 降级对比: MLIR 5个ins + sqrt, Triton 5行组合
# bishengir: 拆为 elemwise+math 组合
import triton
import triton.language as tl
import torch

@triton.jit
def bn_norm_kernel(X, Y, MU, VAR, GAMMA, BETA, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    mu = tl.load(MU + pid)
    var = tl.load(VAR + pid)
    gamma = tl.load(GAMMA + pid)
    beta = tl.load(BETA + pid)
    y = gamma * (x - mu) / tl.sqrt(var + 1e-5) + beta
    tl.store(Y + offs, y, mask=mask)

if __name__ == "__main__":
    x = torch.randn(16, device='cuda') * 2 + 1
    y = torch.empty(16, device='cuda')
    mu = x.mean().expand(1)
    var = x.var(unbiased=False).expand(1)
    gamma = torch.ones(1, device='cuda')
    beta = torch.zeros(1, device='cuda')
    bn_norm_kernel[(1,)](x, y, mu, var, gamma, beta, N=16, BLOCK=16)
    print(f"✅ BN_norm: mean={y.mean().item():.4f}, std={y.std(unbiased=False).item():.4f}")