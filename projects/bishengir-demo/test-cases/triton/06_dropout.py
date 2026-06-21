# (对应 01_basic/06_dropout.mlir) ⭐
# 公式: y = x * scale (简化版)
# 一句话: 训练时随机让部分神经元"翘课"
# 专业角色: 正则化技术, 防止过拟合. BERT 使用, 现代 LLM 趋向不用
# 用在哪: 全连接层 / Transformer 训练阶段
# 降级对比: MLIR arith.mulf, Triton x*scale
# bishengir: hfusion.elemwise_binary {fun = mul}
import triton
import triton.language as tl
import torch

@triton.jit
def dropout_kernel(X, Y, scale: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = x * scale
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.randn(128, device='cuda')
    y = torch.empty(128, device='cuda')
    dropout_kernel[(4,)](x, y, scale=1.25, BLOCK=32)
    print(f"✅ Dropout: mean(x)={x.mean().item():.4f}, mean(y)={(y/1.25).mean().item():.4f}")