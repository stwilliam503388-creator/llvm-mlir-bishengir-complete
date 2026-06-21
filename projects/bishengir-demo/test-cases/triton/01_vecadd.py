# (对应 01_basic/01_vecadd.mlir) ⭐
# 公式: C[i] = A[i] + B[i]
# 一句话: 两个数组对应位置逐元素相加
# 专业角色: 残差连接 (Residual Connection), y = x + F(x), 每层输出与输入直接相加, 解决深层梯度消失
# 用在哪: Transformer 残差连接 / ResNet shortcut / 几乎全部模型
# 降级对比: MLIR 3行 linalg.generic → 38行LLVM; Triton 1行 a+b
# bishengir: hfusion.elemwise_binary {fun = add}
import triton
import triton.language as tl
import torch

@triton.jit
def vecadd_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    c = a + b  # arith.addf
    tl.store(C + offsets, c, mask=mask)

if __name__ == "__main__":
    N = 128
    A = torch.randn(N, device='cuda', dtype=torch.float16)
    B = torch.randn(N, device='cuda', dtype=torch.float16)
    C = torch.empty(N, device='cuda', dtype=torch.float16)
    vecadd_kernel[(N // 32,)](A, B, C, N, BLOCK=32)
    torch.testing.assert_close(C, A + B)
    print(f"✅ vecadd: C[0]={C[0].item():.4f}")