# (对应 02_intermediate/06_prelu.mlir) ⭐⭐
# 公式: y = x if x > 0 else alpha*x (alpha 可训练)
# 一句话: 让 AI 自己学负数区斜率
# 专业角色: 可学习参数激活, 超分辨率模型常用
# 用在哪: 图像超分辨率 / 精细图像任务
# 降级对比: MLIR mulf+cmpf+select, Triton tl.where()
# bishengir: 条件分支映射
import triton
import triton.language as tl
import torch

@triton.jit
def prelu_kernel(X, Y, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.where(x > 0, x, alpha * x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
    alpha = 0.25
    y = torch.empty(5, device='cuda')
    prelu_kernel[(1,)](x, y, alpha, BLOCK=32)
    expected = torch.where(x > 0, x, alpha * x)
    torch.testing.assert_close(y, expected)
    print(f"✅ PReLU(alpha={alpha}): {y.tolist()}")