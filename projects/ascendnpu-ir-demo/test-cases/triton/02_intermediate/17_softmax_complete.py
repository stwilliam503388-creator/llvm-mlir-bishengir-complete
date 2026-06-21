1|# (对应 mlir/02_intermediate/09_softmax_complete.mlir) ⭐⭐
2|# 公式: y = e^{x - max(x)}
3|# 一句话: 先减最大值再取指数, 防止溢出
4|# 专业角色: Attention 机制核心操作, 数值稳定版
5|# 用在哪: Transformer / BERT/GPT Attention 层
6|# 降级对比: MLIR alloc+reduce+broadcast+exp, Triton 1行
7|# bishengir: 组合 reduce+broadcast+elemwise
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def softmax_stable_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    x_max = tl.max(x, axis=0)
18|    y = tl.exp(x - x_max)
19|    tl.store(Y + offsets, y)
20|
21|if __name__ == "__main__":
22|    x = torch.tensor([1000.0, 999.0, 998.0], device='cuda')
23|    y = torch.empty(3, device='cuda')
24|    softmax_stable_kernel[(1,)](x, y, BLOCK=32)
25|    print(f"✅ SoftmaxStable: {y.tolist()} (no overflow!)")