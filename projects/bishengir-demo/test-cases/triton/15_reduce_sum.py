# (对应 02_intermediate/07_reduce_sum.mlir) ⭐⭐
# 公式: sum = sum_i x[i], 归约
# 一句话: 把一堆数加成一个数
# 专业角色: LayerNorm 分母, Softmax 分母, 聚合操作基础
# 用在哪: LayerNorm / Softmax / 各种聚合
# 降级对比: MLIR reduction iterator, Triton tl.sum(x)
# bishengir: hfusion.reduce {fun = add}
import triton
import triton.language as tl
import torch

@triton.jit
def reduce_sum_kernel(X, SUM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    s = tl.sum(x, axis=0)
    tl.store(SUM, s)

if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda').flatten()
    s = torch.zeros(1, device='cuda')
    reduce_sum_kernel[(1,)](x, s, BLOCK=4)
    print(f"✅ ReduceSum: sum={s.item():.1f}, expected=10.0")