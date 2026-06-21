# (对应 02_intermediate/09_softmax_complete.mlir) ⭐⭐
# 公式: y = e^{x - max(x)}
# 一句话: 先减最大值再取指数, 防止溢出
# 专业角色: Attention 机制核心操作, 数值稳定版
# 用在哪: Transformer / BERT/GPT Attention 层
# 降级对比: MLIR alloc+reduce+broadcast+exp, Triton 1行
# bishengir: 组合 reduce+broadcast+elemwise
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_stable_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    x_max = tl.max(x, axis=0)
    y = tl.exp(x - x_max)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([1000.0, 999.0, 998.0], device='cuda')
    y = torch.empty(3, device='cuda')
    softmax_stable_kernel[(1,)](x, y, BLOCK=32)
    print(f"✅ SoftmaxStable: {y.tolist()} (no overflow!)")