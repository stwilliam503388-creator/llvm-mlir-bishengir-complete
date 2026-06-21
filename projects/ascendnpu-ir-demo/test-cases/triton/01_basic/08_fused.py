1|# (对应 mlir/01_basic/08_fused.mlir) ⭐
2|# 操作: C = A + B; D = C * A (两个操作融合)
3|# 一句话: 两步并为一步, 减少中间内存读写
4|# 专业角色: Kernel Fusion 演示, TVM/XLA/Triton 核心优化
5|# 用在哪: 编译器优化概念演示
6|# 降级对比: MLIR 2次 linalg.generic, Triton 1个 kernel
7|# bishengir: HFusion 算子融合概念
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def fused_kernel(A, B, D, N: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    mask = offsets < N
17|    a = tl.load(A + offsets, mask=mask)
18|    b = tl.load(B + offsets, mask=mask)
19|    c = a + b  # 第一步
20|    d = c * a  # 第二步 (不写回内存, 直接算)
21|    tl.store(D + offsets, d, mask=mask)
22|
23|if __name__ == "__main__":
24|    N = 128
25|    A = torch.randn(N, device='cuda')
26|    B = torch.randn(N, device='cuda')
27|    D = torch.empty(N, device='cuda')
28|    fused_kernel[(N // 32,)](A, B, D, N, BLOCK=32)
29|    expected = (A + B) * A
30|    torch.testing.assert_close(D, expected)
31|    print(f"✅ Fused: D[0]={D[0].item():.4f}, expected={expected[0].item():.4f}")