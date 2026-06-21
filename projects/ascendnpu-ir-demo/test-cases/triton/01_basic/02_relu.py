1|# (对应 mlir/01_basic/02_relu.mlir) ⭐
2|# 公式: y = max(0, x)
3|# 一句话: 负数变0, 正数不变
4|# 专业角色: 最常用激活函数, 引入非线性, 计算成本极低 (一次比较)
5|# 用在哪: ResNet/VGG/CNN 全系列, LLM 早期使用
6|# 降级对比: MLIR arith.cmpf+select, Triton tl.maximum(x,0)
7|# bishengir: hfusion.elemwise_unary {fun = relu}
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def relu_kernel(X, Y, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    x = tl.load(X + offsets)
17|    y = tl.maximum(x, 0.0)  # ReLU
18|    tl.store(Y + offsets, y)
19|
20|if __name__ == "__main__":
21|    x = torch.randn(128, device='cuda') - 0.5
22|    y = torch.empty(128, device='cuda')
23|    relu_kernel[(4,)](x, y, BLOCK=32)
24|    torch.testing.assert_close(y, torch.relu(x))
25|    print(f"✅ ReLU: negative={y[y<0].numel()}, pos={y[y>0].numel()}")