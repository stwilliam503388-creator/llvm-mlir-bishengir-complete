1|# (对应 mlir/02_intermediate/06_prelu.mlir) ⭐⭐
2|# 公式: y = x if x > 0 else alpha*x (alpha 可训练)
3|# 一句话: 让 AI 自己学负数区斜率
4|# 专业角色: 可学习参数激活, 超分辨率模型常用
5|# 用在哪: 图像超分辨率 / 精细图像任务
6|# 降级对比: MLIR mulf+cmpf+select, Triton tl.where()
7|# bishengir: 条件分支映射
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def prelu_kernel(X, Y, alpha, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.where(x > 0, x, alpha * x)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
22|    alpha = 0.25
23|    y = torch.empty(5, device='cuda')
24|    prelu_kernel[(1,)](x, y, alpha, BLOCK=32)
25|    expected = torch.where(x > 0, x, alpha * x)
26|    torch.testing.assert_close(y, expected)
27|    print(f"✅ PReLU(alpha={alpha}): {y.tolist()}")