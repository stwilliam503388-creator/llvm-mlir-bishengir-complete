1|# (对应 mlir/02_intermediate/03_leaky_relu.mlir) ⭐⭐
2|# 公式: y = x if x > 0 else 0.01x
3|# 一句话: ReLU 改进版, 负数区留一小缝
4|# 专业角色: 解决 ReLU 死亡问题, GAN 标配
5|# 用在哪: GAN / 部分传统 CNN
6|# 降级对比: MLIR cmpf+mulf+select, Triton tl.where(x>0,x,0.01*x)
7|# bishengir: 条件分支映射
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def leaky_relu_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.where(x > 0, x, 0.01 * x)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
22|    y = torch.empty(5, device='cuda')
23|    leaky_relu_kernel[(1,)](x, y, BLOCK=32)
24|    expected = torch.where(x > 0, x, 0.01 * x)
25|    torch.testing.assert_close(y, expected)
26|    print(f"✅ LeakyReLU: {dict(zip(x.cpu().tolist(), [f'{v:.4f}' for v in y.cpu().tolist()]))}")