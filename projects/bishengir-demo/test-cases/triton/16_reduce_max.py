# (对应 02_intermediate/08_reduce_max.mlir) ⭐⭐
# 公式: max_x = max(x_i)
# 一句话: 从一堆数里找出最大的
# 专业角色: Softmax 数值稳定关键步骤, 减 max 防 exp 溢出
# 用在哪: Softmax 数值稳定 / Max Pooling
# 降级对比: MLIR reduction+cmpf+select, Triton tl.max(x)
# bishengir: hfusion.reduce {fun = max}
import triton
import triton.language as tl
import torch

@triton.jit
def reduce_max_kernel(X, MAX, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    m = tl.max(x, axis=0)
    tl.store(MAX, m)

if __name__ == "__main__":
    x = torch.tensor([1.0, 5.0, 2.0, 8.0], device='cuda')
    m = torch.zeros(1, device='cuda')
    reduce_max_kernel[(1,)](x, m, BLOCK=4)
    print(f"✅ ReduceMax: max={m.item():.1f}, expected=8.0")