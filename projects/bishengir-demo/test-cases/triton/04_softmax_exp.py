# (对应 01_basic/04_softmax_exp.mlir) ⭐
# 公式: y = e^{x}
# 一句话: 把分数换算成正数, 放大差距
# 专业角色: Attention 机制核心: softmax(QK^T / sqrt(d))
# 用在哪: Transformer Attention / 多分类输出层
# 降级对比: MLIR math.exp, Triton tl.exp(x)
# bishengir: 逐元素 math 操作映射
import triton
import triton.language as tl
import torch

@triton.jit
def exp_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.exp(x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.empty(3, device='cuda')
    exp_kernel[(1,)](x, y, BLOCK=32)
    torch.testing.assert_close(y, torch.exp(x))
    print(f"✅ Softmax(exp): {y.tolist()}")