# (对应 02_intermediate/03_leaky_relu.mlir) ⭐⭐
# 公式: y = x if x > 0 else 0.01x
# 一句话: ReLU 改进版, 负数区留一小缝
# 专业角色: 解决 ReLU 死亡问题, GAN 标配
# 用在哪: GAN / 部分传统 CNN
# 降级对比: MLIR cmpf+mulf+select, Triton tl.where(x>0,x,0.01*x)
# bishengir: 条件分支映射
import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.where(x > 0, x, 0.01 * x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
    y = torch.empty(5, device='cuda')
    leaky_relu_kernel[(1,)](x, y, BLOCK=32)
    expected = torch.where(x > 0, x, 0.01 * x)
    torch.testing.assert_close(y, expected)
    print(f"✅ LeakyReLU: {dict(zip(x.cpu().tolist(), [f'{v:.4f}' for v in y.cpu().tolist()]))}")