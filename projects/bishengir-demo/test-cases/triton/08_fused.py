# Fused — 算子融合 (对应 01_basic/08_fused.mlir)
# MLIR: 连续两次 linalg.generic (addf + mulf)
# Triton: 在同一个 kernel 里做完两步, 无需中间 buffer
# 公式: C = A + B; D = C * A

import triton
import triton.language as tl
import torch

@triton.jit
def fused_kernel(A, B, D, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    c = a + b  # 第一步
    d = c * a  # 第二步 (不写回内存, 直接算)
    tl.store(D + offsets, d, mask=mask)

if __name__ == "__main__":
    N = 128
    A = torch.randn(N, device='cuda')
    B = torch.randn(N, device='cuda')
    D = torch.empty(N, device='cuda')
    fused_kernel[(N // 32,)](A, B, D, N, BLOCK=32)
    expected = (A + B) * A
    torch.testing.assert_close(D, expected)
    print(f"✅ Fused: D[0]={D[0].item():.4f}, expected={expected[0].item():.4f}")
