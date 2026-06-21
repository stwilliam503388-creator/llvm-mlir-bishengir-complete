# (对应 01_basic/05_broadcast.mlir) ⭐
# 公式: B[i][j] = A, 标量到矩阵
# 一句话: 把1个数复制到整个数组
# 专业角色: 张量维度自动扩展, bias 加法 / 归一化参数广播
# 用在哪: 所有带 bias/归一化的层
# 降级对比: MLIR affine_map<(i,j)->()>, Triton 自动广播
# bishengir: hfusion.broadcast
import triton
import triton.language as tl
import torch

@triton.jit
def broadcast_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(X)  # 标量, 自动广播
    tl.store(Y + offsets, a, mask=mask)

if __name__ == "__main__":
    N = 16
    x = torch.tensor([3.14], device='cuda')
    y = torch.empty(N, device='cuda')
    broadcast_kernel[(N // 4,)](x, y, N, BLOCK=4)
    expected = torch.full([N], 3.14, device='cuda')
    torch.testing.assert_close(y, expected)
    print(f"✅ Broadcast: {y[:5].tolist()}...")