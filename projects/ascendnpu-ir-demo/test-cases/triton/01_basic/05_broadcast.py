1|# (对应 mlir/01_basic/05_broadcast.mlir) ⭐
2|# 公式: B[i][j] = A, 标量到矩阵
3|# 一句话: 把1个数复制到整个数组
4|# 专业角色: 张量维度自动扩展, bias 加法 / 归一化参数广播
5|# 用在哪: 所有带 bias/归一化的层
6|# 降级对比: MLIR affine_map<(i,j)->()>, Triton 自动广播
7|# bishengir: hfusion.broadcast
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def broadcast_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    mask = offsets < N
17|    a = tl.load(X)  # 标量, 自动广播
18|    tl.store(Y + offsets, a, mask=mask)
19|
20|if __name__ == "__main__":
21|    N = 16
22|    x = torch.tensor([3.14], device='cuda')
23|    y = torch.empty(N, device='cuda')
24|    broadcast_kernel[(N // 4,)](x, y, N, BLOCK=4)
25|    expected = torch.full([N], 3.14, device='cuda')
26|    torch.testing.assert_close(y, expected)
27|    print(f"✅ Broadcast: {y[:5].tolist()}...")