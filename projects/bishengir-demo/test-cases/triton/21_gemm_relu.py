# GEMM + ReLU 融合 (对应 03_advanced/02_gemm_relu.mlir)
# MLIR: linalg.matmul + linalg.generic
# Triton: 在同一个 kernel 里做 matmul + activation
# 公式: y = ReLU(x @ W)

import triton
import triton.language as tl
import torch

@triton.jit
def gemm_relu_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    c = tl.maximum(c, 0.0)  # ReLU 融合在 kernel 内
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)

if __name__ == "__main__":
    A = torch.randn(64, 64, device='cuda')
    B = torch.randn(64, 64, device='cuda')
    C = torch.empty(64, 64, device='cuda')
    grid = (64 // 16, 64 // 16)
    gemm_relu_kernel[grid](A, B, C, 64, 64, 64, BLOCK=16)
    expected = torch.relu(A @ B)
    max_err = (C - expected).abs().max().item()
    print(f"✅ GEMM+ReLU: max_error={max_err:.6f}, neg_count={(C<0).sum().item()}")
