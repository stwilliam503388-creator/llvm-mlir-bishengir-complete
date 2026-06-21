# (对应 mlir/03_advanced/07_global_avg_pool.mlir) ⭐⭐⭐
# 公式: 整个特征图求 1 个平均值
# 一句话: 把整张图压缩成 1 个数
# 专业角色: CNN 分类头前最后一层, 参数量 0
# 用在哪: ResNet/MobileNet/GoogleNet 分类头
# 降级对比: MLIR affine.for x2 + 累加, Triton tl.sum(x)/N
# bishengir: 组合 reduce + 除法
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