# (对应 02_intermediate/05_hard_sigmoid.mlir) ⭐⭐
# 公式: hard_sigmoid(x) = clamp(0.2x+0.5, 0, 1)
# 一句话: 用折线近似 S 形, 省去 exp 计算
# 专业角色: 轻量化模型激活, 计算量小 3 倍
# 用在哪: MobileNetV3 / 轻量化 CNN
# 降级对比: MLIR maximumf+minimumf, Triton tl.clamp()
# bishengir: 分段线性函数映射
import triton
import triton.language as tl
import torch

@triton.jit
def hard_sigmoid_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.clamp(0.2 * x + 0.5, 0.0, 1.0)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-5.0, -2.5, 0.0, 2.5, 5.0], device='cuda')
    y = torch.empty(5, device='cuda')
    hard_sigmoid_kernel[(1,)](x, y, BLOCK=32)
    expected = torch.clamp(0.2 * x + 0.5, 0, 1)
    torch.testing.assert_close(y, expected)
    print(f"✅ HardSigmoid: {y.tolist()}")