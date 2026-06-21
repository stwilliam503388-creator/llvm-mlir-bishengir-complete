1|# (对应 mlir/02_intermediate/01_sigmoid.mlir) ⭐⭐
2|# 公式: sigma(x) = 1 / (1 + e^{-x}), 输出 (0, 1)
3|# 一句话: 把任意数压缩到 0~1 之间
4|# 专业角色: 二分类输出层, RNN 门控, SwiGLU (LLaMA) 用 sigmoid 做门控
5|# 用在哪: 二分类 / RNN门控 / SwiGLU 中间步骤
6|# 降级对比: MLIR 4步 (negf+exp+addf+divf), Triton 1步 tl.sigmoid(x)
7|# bishengir: 组合后可融合
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def sigmoid_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.sigmoid(x)
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
22|    y = torch.empty(5, device='cuda')
23|    sigmoid_kernel[(1,)](x, y, BLOCK=32)
24|    print(f"✅ Sigmoid: {dict(zip(x.cpu().tolist(), [f'{v:.4f}' for v in y.cpu().tolist()]))}")