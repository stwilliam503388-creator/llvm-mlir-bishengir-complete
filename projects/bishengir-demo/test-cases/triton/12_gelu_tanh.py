# (对应 02_intermediate/04_gelu_tanh.mlir) ⭐⭐
# 公式: gelu(x) ~= 0.5*x*(1+tanh(x))
# 一句话: 平滑版 ReLU, 负数区逐渐变0
# 专业角色: BERT/GPT-2/GPT-3 FFN 层标准激活函数
# 用在哪: BERT/RoBERTa/GPT-2/GPT-3 FFN 层
# 降级对比: MLIR 4步 (tanh+addf+mulf x2), Triton 1行
# bishengir: 组合后可融合
import triton
import triton.language as tl
import torch

@triton.jit
def gelu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    gelu = 0.5 * x * (1.0 + tl.tanh(x))
    tl.store(Y + offsets, gelu)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda')
    y = torch.empty(128, device='cuda')
    gelu_kernel[(4,)](x, y, BLOCK=32)
    expected = 0.5 * x * (1.0 + torch.tanh(x))
    torch.testing.assert_close(y, expected)
    print(f"✅ GELU: x[0]={x[0].item():.4f}, gelu={y[0].item():.4f}")