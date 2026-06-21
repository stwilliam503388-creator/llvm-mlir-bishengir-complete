1|# (对应 mlir/02_intermediate/10_clamp.mlir) ⭐⭐
2|# 公式: y = clamp(x, -1, 1)
3|# 一句话: 超过范围的值截断到边界
4|# 专业角色: 梯度裁剪 (LLM 训练), 量化前处理
5|# 用在哪: 梯度裁剪 / 量化前处理 / 强化学习
6|# 降级对比: MLIR cmpf+select x2, Triton tl.clamp(x,-1,1)
7|# bishengir: 分段线性函数映射
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def clamp_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.clamp(x, -1.0, 1.0)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([-2.0, -1.5, 0.0, 0.5, 2.0], device='cuda')
22|    y = torch.empty(5, device='cuda')
23|    clamp_kernel[(1,)](x, y, BLOCK=5)
24|    expected = torch.clamp(x, -1, 1)
25|    torch.testing.assert_close(y, expected)
26|    print(f"✅ Clamp: {y.tolist()}")