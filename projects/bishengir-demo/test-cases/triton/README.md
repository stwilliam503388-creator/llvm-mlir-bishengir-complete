# Triton 代码示例

> 28 个 MLIR 测试用例反向生成的 Triton Python 代码。
> 每个 `.py` 文件对应 `test-cases/` 下同编号的 `.mlir` 文件。

## 对应关系

| 级别 | MLIR 文件 | Triton 文件 | 操作 |
|------|----------|------------|------|
| 01_basic | 01_vecadd.mlir | 01_vecadd.py | 向量加法 (残差连接) |
| 01_basic | 02_relu.mlir | 02_relu.py | ReLU 激活 |
| 01_basic | 03_tanh.mlir | 03_tanh.py | Tanh 激活 |
| 01_basic | 04_softmax_exp.mlir | 04_softmax_exp.py | Softmax 指数部分 |
| 01_basic | 05_broadcast.mlir | 05_broadcast.py | 标量广播 |
| 01_basic | 06_dropout.mlir | 06_dropout.py | Dropout (简化) |
| 01_basic | 07_fill.mlir | 07_fill.py | 常量填充 |
| 01_basic | 08_fused.mlir | 08_fused.py | 算子融合 (add+mul) |
| 02_intermediate | 01_sigmoid.mlir | 09_sigmoid.py | Sigmoid 激活 |
| 02_intermediate | 02_silu.mlir | 10_silu.py | SiLU (LLaMA) |
| 02_intermediate | 03_leaky_relu.mlir | 11_leaky_relu.py | LeakyReLU |
| 02_intermediate | 04_gelu_tanh.mlir | 12_gelu_tanh.py | GELU (BERT) |
| 02_intermediate | 05_hard_sigmoid.mlir | 13_hard_sigmoid.py | Hard Sigmoid |
| 02_intermediate | 06_prelu.mlir | 14_prelu.py | PReLU |
| 02_intermediate | 07_reduce_sum.mlir | 15_reduce_sum.py | 归约求和 |
| 02_intermediate | 08_reduce_max.mlir | 16_reduce_max.py | 归约最大值 |
| 02_intermediate | 09_softmax_complete.mlir | 17_softmax_complete.py | Softmax 数值稳定 |
| 02_intermediate | 10_clamp.mlir | 18_clamp.py | 数值裁剪 |
| 02_intermediate | 11_layer_norm.mlir | 19_layer_norm.py | Layer Norm |
| 03_advanced | 01_matmul.mlir | 20_matmul.py | 矩阵乘法 |
| 03_advanced | 02_gemm_relu.mlir | 21_gemm_relu.py | GEMM+ReLU 融合 |
| 03_advanced | 04_conv2d.mlir | 22_conv2d.py | 二维卷积 |
| 03_advanced | 05_max_pool.mlir | 23_max_pool.py | 最大池化 |
| 03_advanced | 06_avg_pool.mlir | 24_avg_pool.py | 平均池化 |
| 03_advanced | 07_global_avg_pool.mlir | 25_global_avg_pool.py | 全局平均池化 |
| 03_advanced | 03_depthwise_conv.mlir | 26_depthwise_conv.py | 深度卷积 |
| 03_advanced | 08_batch_norm_part1.mlir | 27_batch_norm_part1.py | BN 均值计算 |
| 03_advanced | 09_batch_norm_part2.mlir | 28_batch_norm_part2.py | BN 标准化 |

## 运行

需要 NVIDIA GPU + CUDA (Triton 不能运行在 Apple Silicon 上)：

```bash
python3 test-cases/triton/20_matmul.py     # 跑矩阵乘法
python3 test-cases/triton/01_vecadd.py      # 跑向量加法
```

## 从 MLIR 到 Triton 的转换要点

| MLIR 概念 | Triton 等价 | 说明 |
|-----------|-------------|------|
| linalg.generic + arith.addf | `a + b` | 逐元素运算直接用 Python 运算符 |
| arith.cmpf + select | `tl.where(cond, a, b)` | 条件分支 |
| linalg.reduce | `tl.sum()` / `tl.max()` | 归约操作 |
| linalg.broadcast | 自动广播 | Triton 自动处理 |
| math.exp / tanh | `tl.exp()` / `tl.tanh()` | 数学函数 |
| linalg.matmul | `tl.dot()` | 矩阵乘法使用 tensor core |
| affine.for | `tl.arange()` + 循环 | 手动循环 |
