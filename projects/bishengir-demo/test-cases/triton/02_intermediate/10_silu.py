1|# (对应 mlir/02_intermediate/02_silu.mlir) ⭐⭐
2|# 公式: silu(x) = x * sigmoid(x)
3|# 一句话: 输入乘它自己的门控值, LLaMA 系列首选激活
4|# 专业角色: SwiGLU = SiLU(x) * y, LLaMA/Mistral/Gemma FFN 层标配
5|# 用在哪: LLaMA/Mistral/Gemma FFN 层
6|# 降级对比: MLIR 5步组合, Triton 1行 x*tl.sigmoid(x)
7|# bishengir: 组合后可融合
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def silu_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = x * tl.sigmoid(x)  # SiLU
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.randn(128, device='cuda')
22|    y = torch.empty(128, device='cuda')
23|    silu_kernel[(4,)](x, y, BLOCK=32)
24|    expected = x * torch.sigmoid(x)
25|    torch.testing.assert_close(y, expected)
26|    print(f"✅ SiLU: x[0]={x[0].item():.4f}, silu={y[0].item():.4f}")