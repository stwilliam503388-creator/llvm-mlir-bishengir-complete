1|# (对应 mlir/01_basic/01_vecadd.mlir) ⭐
2|# 公式: C[i] = A[i] + B[i]
3|# 一句话: 两个数组对应位置逐元素相加
4|# 专业角色: 残差连接 (Residual Connection), y = x + F(x), 每层输出与输入直接相加, 解决深层梯度消失
5|# 用在哪: Transformer 残差连接 / ResNet shortcut / 几乎全部模型
6|# 降级对比: MLIR 3行 linalg.generic → 38行LLVM; Triton 1行 a+b
7|# bishengir: hfusion.elemwise_binary {fun = add}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def vecadd_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    mask = offsets < N
17|    a = tl.load(A + offsets, mask=mask)
18|    b = tl.load(B + offsets, mask=mask)
19|    c = a + b  # arith.addf
20|    tl.store(C + offsets, c, mask=mask)
21|
22|if __name__ == "__main__":
23|    N = 128
24|    A = torch.randn(N, device='cuda', dtype=torch.float16)
25|    B = torch.randn(N, device='cuda', dtype=torch.float16)
26|    C = torch.empty(N, device='cuda', dtype=torch.float16)
27|    vecadd_kernel[(N // 32,)](A, B, C, N, BLOCK=32)
28|    torch.testing.assert_close(C, A + B)
29|    print(f"✅ vecadd: C[0]={C[0].item():.4f}")