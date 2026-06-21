1|# (对应 mlir/03_advanced/02_gemm_relu.mlir) ⭐⭐⭐
2|# 公式: y = ReLU(x @ W)
3|# 一句话: 矩阵乘完立刻激活, 两步并一步
4|# 专业角色: MLP 层标准融合模式, 减少中间 buffer 读写
5|# 用在哪: MLP 层 / FFN 层
6|# 降级对比: MLIR 2阶段 pipeline, Triton 1个 kernel 内完成
7|# bishengir: 可融合为单个 kernel
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def gemm_relu_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
14|    pid_m = tl.program_id(0)
15|    pid_n = tl.program_id(1)
16|    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
17|    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
18|    offs_k = tl.arange(0, BLOCK)
19|    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
20|    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
21|    c = tl.dot(a, b)
22|    c = tl.maximum(c, 0.0)  # ReLU 融合在 kernel 内
23|    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)
24|
25|if __name__ == "__main__":
26|    A = torch.randn(64, 64, device='cuda')
27|    B = torch.randn(64, 64, device='cuda')
28|    C = torch.empty(64, 64, device='cuda')
29|    grid = (64 // 16, 64 // 16)
30|    gemm_relu_kernel[grid](A, B, C, 64, 64, 64, BLOCK=16)
31|    expected = torch.relu(A @ B)
32|    max_err = (C - expected).abs().max().item()
33|    print(f"✅ GEMM+ReLU: max_error={max_err:.6f}, neg_count={(C<0).sum().item()}")