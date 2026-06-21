1|# (对应 mlir/02_intermediate/08_reduce_max.mlir) ⭐⭐
2|# 公式: max_x = max(x_i)
3|# 一句话: 从一堆数里找出最大的
4|# 专业角色: Softmax 数值稳定关键步骤, 减 max 防 exp 溢出
5|# 用在哪: Softmax 数值稳定 / Max Pooling
6|# 降级对比: MLIR reduction+cmpf+select, Triton tl.max(x)
7|# bishengir: hfusion.reduce {fun = max}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def reduce_max_kernel(X, MAX, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    m = tl.max(x, axis=0)
18|    tl.store(MAX, m)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([1.0, 5.0, 2.0, 8.0], device='cuda')
22|    m = torch.zeros(1, device='cuda')
23|    reduce_max_kernel[(1,)](x, m, BLOCK=4)
24|    print(f"✅ ReduceMax: max={m.item():.1f}, expected=8.0")