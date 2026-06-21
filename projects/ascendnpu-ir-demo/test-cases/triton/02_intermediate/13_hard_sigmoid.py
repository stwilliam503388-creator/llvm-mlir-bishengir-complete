1|# (对应 mlir/02_intermediate/05_hard_sigmoid.mlir) ⭐⭐
2|# 公式: hard_sigmoid(x) = clamp(0.2x+0.5, 0, 1)
3|# 一句话: 用折线近似 S 形, 省去 exp 计算
4|# 专业角色: 轻量化模型激活, 计算量小 3 倍
5|# 用在哪: MobileNetV3 / 轻量化 CNN
6|# 降级对比: MLIR maximumf+minimumf, Triton tl.clamp()
7|# bishengir: 分段线性函数映射
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def hard_sigmoid_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.clamp(0.2 * x + 0.5, 0.0, 1.0)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([-5.0, -2.5, 0.0, 2.5, 5.0], device='cuda')
22|    y = torch.empty(5, device='cuda')
23|    hard_sigmoid_kernel[(1,)](x, y, BLOCK=32)
24|    expected = torch.clamp(0.2 * x + 0.5, 0, 1)
25|    torch.testing.assert_close(y, expected)
26|    print(f"✅ HardSigmoid: {y.tolist()}")