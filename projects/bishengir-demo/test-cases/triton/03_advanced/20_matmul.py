1|# (对应 mlir/03_advanced/01_matmul.mlir) ⭐⭐⭐
2|# 公式: C = A @ B, 4x4 x 4x4 -> 4x4
3|# 一句话: 深度学习最核心操作, 占 LLM 算力 60-80%
4|# 专业角色: Linear 层 y=xW^T+b, Attention Q/K/V 投影, FFN up/down
5|# 用在哪: Linear / Attention / FFN, 无处不在
6|# 降级对比: MLIR 1行 linalg.matmul -> 74行LLVM; Triton 1行 tl.dot()
7|# bishengir: hfusion.cube_matmul -> hivm.mmul (1行NPU指令)
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def matmul_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
14|    pid_m = tl.program_id(0)
15|    pid_n = tl.program_id(1)
16|    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
17|    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
18|    offs_k = tl.arange(0, BLOCK)
19|    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
20|    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
21|    c = tl.dot(a, b)
22|    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)
23|
24|if __name__ == "__main__":
25|    A = torch.randn(64, 64, device='cuda')
26|    B = torch.randn(64, 64, device='cuda')
27|    C = torch.empty(64, 64, device='cuda')
28|    grid = (64 // 16, 64 // 16)
29|    matmul_kernel[grid](A, B, C, 64, 64, 64, BLOCK=16)
30|    expected = A @ B
31|    max_err = (C - expected).abs().max().item()
32|    print(f"✅ MatMul: max_error={max_err:.6f}")