1|# (对应 mlir/02_intermediate/07_reduce_sum.mlir) ⭐⭐
2|# 公式: sum = sum_i x[i], 归约
3|# 一句话: 把一堆数加成一个数
4|# 专业角色: LayerNorm 分母, Softmax 分母, 聚合操作基础
5|# 用在哪: LayerNorm / Softmax / 各种聚合
6|# 降级对比: MLIR reduction iterator, Triton tl.sum(x)
7|# bishengir: hfusion.reduce {fun = add}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def reduce_sum_kernel(X, SUM, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    s = tl.sum(x, axis=0)
18|    tl.store(SUM, s)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda').flatten()
22|    s = torch.zeros(1, device='cuda')
23|    reduce_sum_kernel[(1,)](x, s, BLOCK=4)
24|    print(f"✅ ReduceSum: sum={s.item():.1f}, expected=10.0")