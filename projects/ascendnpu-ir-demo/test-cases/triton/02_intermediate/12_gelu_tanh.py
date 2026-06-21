1|# (对应 mlir/02_intermediate/04_gelu_tanh.mlir) ⭐⭐
2|# 公式: gelu(x) ~= 0.5*x*(1+tanh(x))
3|# 一句话: 平滑版 ReLU, 负数区逐渐变0
4|# 专业角色: BERT/GPT-2/GPT-3 FFN 层标准激活函数
5|# 用在哪: BERT/RoBERTa/GPT-2/GPT-3 FFN 层
6|# 降级对比: MLIR 4步 (tanh+addf+mulf x2), Triton 1行
7|# bishengir: 组合后可融合
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def gelu_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    gelu = 0.5 * x * (1.0 + tl.tanh(x))
18|    tl.store(Y + offsets, gelu)
19|
20|if __name__ == "__main__":
21|    x = torch.randn(128, device='cuda')
22|    y = torch.empty(128, device='cuda')
23|    gelu_kernel[(4,)](x, y, BLOCK=32)
24|    expected = 0.5 * x * (1.0 + torch.tanh(x))
25|    torch.testing.assert_close(y, expected)
26|    print(f"✅ GELU: x[0]={x[0].item():.4f}, gelu={y[0].item():.4f}")