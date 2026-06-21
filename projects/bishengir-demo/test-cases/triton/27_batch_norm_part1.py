# (对应 03_advanced/08_batch_norm_part1.mlir) ⭐⭐⭐
# 公式: mean[j] = sum_i x[i][j] / N
# 一句话: 算一批数据在各通道的平均水平
# 专业角色: BN 第一步, 稳定训练, 允许高学习率
# 用在哪: ResNet/CNN 全系 (训练阶段)
# 降级对比: MLIR reduction+parallel, Triton tl.sum(x)/N
# bishengir: 拆为 reduce+broadcast+elemwise 三步
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