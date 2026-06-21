# Broadcast — 广播 (对应 01_basic/05_broadcast.mlir)
# MLIR: affine_map<(i,j)->()>
# Triton: 直接使用标量, 自动广播
# 公式: B[i][j] = A (标量到矩阵)

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
