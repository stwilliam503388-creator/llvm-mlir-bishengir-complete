1|# (对应 mlir/01_basic/06_dropout.mlir) ⭐
2|# 公式: y = x * scale (简化版)
3|# 一句话: 训练时随机让部分神经元"翘课"
4|# 专业角色: 正则化技术, 防止过拟合. BERT 使用, 现代 LLM 趋向不用
5|# 用在哪: 全连接层 / Transformer 训练阶段
6|# 降级对比: MLIR arith.mulf, Triton x*scale
7|# bishengir: hfusion.elemwise_binary {fun = mul}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def dropout_kernel(X, Y, scale: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = x * scale
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.randn(128, device='cuda')
22|    y = torch.empty(128, device='cuda')
23|    dropout_kernel[(4,)](x, y, scale=1.25, BLOCK=32)
24|    print(f"✅ Dropout: mean(x)={x.mean().item():.4f}, mean(y)={(y/1.25).mean().item():.4f}")