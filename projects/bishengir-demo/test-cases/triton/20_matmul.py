# (对应 03_advanced/01_matmul.mlir) ⭐⭐⭐
# 公式: C = A @ B, 4x4 x 4x4 -> 4x4
# 一句话: 深度学习最核心操作, 占 LLM 算力 60-80%
# 专业角色: Linear 层 y=xW^T+b, Attention Q/K/V 投影, FFN up/down
# 用在哪: Linear / Attention / FFN, 无处不在
# 降级对比: MLIR 1行 linalg.matmul -> 74行LLVM; Triton 1行 tl.dot()
# bishengir: hfusion.cube_matmul -> hivm.mmul (1行NPU指令)
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