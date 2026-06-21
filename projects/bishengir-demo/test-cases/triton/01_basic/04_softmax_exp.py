1|# (对应 mlir/01_basic/04_softmax_exp.mlir) ⭐
2|# 公式: y = e^{x}
3|# 一句话: 把分数换算成正数, 放大差距
4|# 专业角色: Attention 机制核心: softmax(QK^T / sqrt(d))
5|# 用在哪: Transformer Attention / 多分类输出层
6|# 降级对比: MLIR math.exp, Triton tl.exp(x)
7|# bishengir: 逐元素 math 操作映射
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def exp_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.exp(x)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
22|    y = torch.empty(3, device='cuda')
23|    exp_kernel[(1,)](x, y, BLOCK=32)
24|    torch.testing.assert_close(y, torch.exp(x))
25|    print(f"✅ Softmax(exp): {y.tolist()}")