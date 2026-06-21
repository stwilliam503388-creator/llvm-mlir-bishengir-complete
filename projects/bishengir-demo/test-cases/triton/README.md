# Triton 代码示例

> 28 个 MLIR 测试用例反向生成的 Triton Python 代码。
> 每文件对应 `test-cases/` 下同编号的 MLIR 文件。

## 完整映射

见 `MAPPING.md` — 包含：每文件的 Triton 代码 ↔ MLIR 代码 ↔ 公式 ↔ AI 角色 ↔ 降级模式对照。

| 级别 | MLIR 路径 | Triton 文件 | 操作 |
|------|----------|------------|------|
| ⭐ basic | `01_basic/01_vecadd.mlir` | `01_vecadd.py` | 向量加法 (残差连接) |
| ⭐ basic | `01_basic/02_relu.mlir` | `02_relu.py` | ReLU 激活 |
| ⭐ basic | `01_basic/03_tanh.mlir` | `03_tanh.py` | Tanh 激活 |
| ⭐ basic | `01_basic/04_softmax_exp.mlir` | `04_softmax_exp.py` | Softmax 指数 |
| ⭐ basic | `01_basic/05_broadcast.mlir` | `05_broadcast.py` | 标量广播 |
| ⭐ basic | `01_basic/06_dropout.mlir` | `06_dropout.py` | Dropout |
| ⭐ basic | `01_basic/07_fill.mlir` | `07_fill.py` | 常量填充 |
| ⭐ basic | `01_basic/08_fused.mlir` | `08_fused.py` | 算子融合 |
| ⭐⭐ intermediate | `02_intermediate/01_sigmoid.mlir` | `09_sigmoid.py` | Sigmoid |
| ⭐⭐ intermediate | `02_intermediate/02_silu.mlir` | `10_silu.py` | SiLU (LLaMA) |
| ⭐⭐ intermediate | `02_intermediate/03_leaky_relu.mlir` | `11_leaky_relu.py` | LeakyReLU |
| ⭐⭐ intermediate | `02_intermediate/04_gelu_tanh.mlir` | `12_gelu_tanh.py` | GELU (BERT) |
| ⭐⭐ intermediate | `02_intermediate/05_hard_sigmoid.mlir` | `13_hard_sigmoid.py` | Hard Sigmoid |
| ⭐⭐ intermediate | `02_intermediate/06_prelu.mlir` | `14_prelu.py` | PReLU |
| ⭐⭐ intermediate | `02_intermediate/07_reduce_sum.mlir` | `15_reduce_sum.py` | 归约求和 |
| ⭐⭐ intermediate | `02_intermediate/08_reduce_max.mlir` | `16_reduce_max.py` | 归约最大值 |
| ⭐⭐ intermediate | `02_intermediate/09_softmax_complete.mlir` | `17_softmax_complete.py` | Softmax 稳定 |
| ⭐⭐ intermediate | `02_intermediate/10_clamp.mlir` | `18_clamp.py` | 数值裁剪 |
| ⭐⭐ intermediate | `02_intermediate/11_layer_norm.mlir` | `19_layer_norm.py` | LayerNorm |
| ⭐⭐⭐ advanced | `03_advanced/01_matmul.mlir` | `20_matmul.py` | 矩阵乘法 |
| ⭐⭐⭐ advanced | `03_advanced/02_gemm_relu.mlir` | `21_gemm_relu.py` | GEMM+ReLU |
| ⭐⭐⭐ advanced | `03_advanced/04_conv2d.mlir` | `22_conv2d.py` | 二维卷积 |
| ⭐⭐⭐ advanced | `03_advanced/05_max_pool.mlir` | `23_max_pool.py` | 最大池化 |
| ⭐⭐⭐ advanced | `03_advanced/06_avg_pool.mlir` | `24_avg_pool.py` | 平均池化 |
| ⭐⭐⭐ advanced | `03_advanced/07_global_avg_pool.mlir` | `25_global_avg_pool.py` | 全局平均池化 |
| ⭐⭐⭐ advanced | `03_advanced/03_depthwise_conv.mlir` | `26_depthwise_conv.py` | 深度卷积 |
| ⭐⭐⭐ advanced | `03_advanced/08_batch_norm_part1.mlir` | `27_batch_norm_part1.py` | BN 均值 |
| ⭐⭐⭐ advanced | `03_advanced/09_batch_norm_part2.mlir` | `28_batch_norm_part2.py` | BN 标准化 |

## 运行

需 NVIDIA GPU + CUDA (Triton 无法在 Apple Silicon 上运行):

```bash
cd projects/bishengir-demo/
python3 test-cases/triton/20_matmul.py   # 矩阵乘
python3 test-cases/triton/01_vecadd.py    # 向量加
python3 test-cases/triton/10_silu.py      # SiLU
```

## MLIR → Triton 对照速查

| MLIR 模式 | Triton 等价 | 说明 |
|-----------|------------|------|
| `linalg.generic + arith.addf` | `a + b` | 逐元素运算法 |
| `arith.cmpf + arith.select` | `tl.where(cond, a, b)` | 条件分支 |
| `math.exp / math.tanh` | `tl.exp() / tl.tanh()` | 数学内建 |
| `linalg.reduce` | `tl.sum() / tl.max()` | 归约函数 |
| `linalg.matmul` | `tl.dot()` | Tensor Core |
| `linalg.broadcast` | 自动广播 | 标量→张量 |

详见 `MAPPING.md`。
