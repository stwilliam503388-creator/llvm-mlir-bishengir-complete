# MatMul — 矩阵乘法 (对应 03_advanced/01_matmul.mlir)
# MLIR: linalg.matmul, 1行->74行LLVM
# Triton: tl.dot, 1行->GPU tensor core
# 公式: C = A @ B, 占LLM算力60-80%

import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)

if __name__ == "__main__":
    A = torch.randn(64, 64, device='cuda')
    B = torch.randn(64, 64, device='cuda')
    C = torch.empty(64, 64, device='cuda')
    grid = (64 // 16, 64 // 16)
    matmul_kernel[grid](A, B, C, 64, 64, 64, BLOCK=16)
    expected = A @ B
    max_err = (C - expected).abs().max().item()
    print(f"✅ MatMul: max_error={max_err:.6f}")
