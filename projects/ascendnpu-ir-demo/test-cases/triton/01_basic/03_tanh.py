1|# (对应 mlir/01_basic/03_tanh.mlir) ⭐
2|# 公式: y = tanh(x), 输出 (-1, 1)
3|# 一句话: 把任意数压缩到 -1~1 之间
4|# 专业角色: RNN/LSTM 门控信号, GELU 近似用到此函数
5|# 用在哪: RNN/LSTM / GELU 中间步骤
6|# 降级对比: MLIR math.tanh, Triton tl.tanh(x)
7|# bishengir: hfusion.elemwise_unary {fun = tanh}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def tanh_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.tanh(x)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.linspace(-5, 5, 128, device='cuda')
22|    y = torch.empty(128, device='cuda')
23|    tanh_kernel[(4,)](x, y, BLOCK=32)
24|    torch.testing.assert_close(y, torch.tanh(x), atol=1e-5, rtol=1e-5)
25|    print(f"✅ Tanh: min={y.min().item():.4f}, max={y.max().item():.4f}")