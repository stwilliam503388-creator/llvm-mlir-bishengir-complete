# MLIR ↔ Triton 映射对照表

每个 MLIR 测试用例对应一个 Triton 代码示例。

> ⭐ = 入门 | ⭐⭐ = 进阶 | ⭐⭐⭐ = 复杂

---

## 01_basic (⭐ 入门) — 10 个

| MLIR | Triton | 操作 | 公式 | MLIR 模式 | Triton 模式 | AI 角色 |
|------|--------|------|------|-----------|------------|---------|
| `01_vecadd.mlir` | `01_vecadd.py` | 向量加法 | C[i]=A[i]+B[i] | `linalg.generic` + `arith.addf` | `a + b` | 残差连接 |
| `02_relu.mlir` | `02_relu.py` | ReLU | max(0,x) | `arith.cmpf` + `select` | `tl.maximum(x,0)` | CNN 激活 |
| `03_tanh.mlir` | `03_tanh.py` | Tanh | (-1,1) | `math.tanh` | `tl.tanh(x)` | RNN 门控 |
| `04_softmax_exp.mlir` | `04_softmax_exp.py` | Softmax 指数 | e^x | `math.exp` | `tl.exp(x)` | Attention |
| `05_broadcast.mlir` | `05_broadcast.py` | 广播 | B[i]=A | `affine_map<()->(i)>` | 自动广播 | bias/Norm |
| `06_dropout.mlir` | `06_dropout.py` | Dropout | x*scale | `arith.mulf` | `x*scale` | 正则化 |
| `07_fill.mlir` | `07_fill.py` | 填充 | A[i]=c | `generic`+yield | `store(val)` | 初始化 |
| `08_fused.mlir` | `08_fused.py` | 融合 | (A+B)*A | 2×`generic` | 1 kernel | 编译优化 |
| `09_vecadd_128.mlir` | `01_vecadd.py` | 向量加法(原始) | C[i]=A[i]+B[i] | 同 `01_vecadd.mlir` | 同 `01_vecadd.py` | 128元素原始版 |
| `10_fused_128.mlir` | `08_fused.py` | 融合(原始) | (A+B)*A | 同 `08_fused.mlir` | 同 `08_fused.py` | 128元素原始版 |

## 02_intermediate (⭐⭐ 进阶)

| MLIR | Triton | 操作 | 公式 | MLIR 模式 | Triton 模式 | AI 角色 |
|------|--------|------|------|-----------|------------|---------|
| `01_sigmoid.mlir` | `09_sigmoid.py` | Sigmoid | 1/(1+e^{-x}) | 4 步组合 | `tl.sigmoid(x)` | 二分类/门控 |
| `02_silu.mlir` | `10_silu.py` | SiLU | x·sigmoid(x) | 5 步组合 | `x*tl.sigmoid(x)` | LLaMA FFN |
| `03_leaky_relu.mlir` | `11_leaky_relu.py` | LeakyReLU | x>0?x:0.01x | `cmpf+mulf+select` | `tl.where()` | GAN |
| `04_gelu_tanh.mlir` | `12_gelu_tanh.py` | GELU | 0.5x(1+tanh(x)) | `tanh+addf+mulf` | `0.5*x*(1+tl.tanh(x))` | BERT/GPT-3 |
| `05_hard_sigmoid.mlir` | `13_hard_sigmoid.py` | HardSigmoid | clamp(0.2x+0.5,0,1) | `max+min` | `tl.clamp()` | MobileNet |
| `06_prelu.mlir` | `14_prelu.py` | PReLU | x>0?x:αx | `mulf+cmpf+select` | `tl.where()` | 超分辨率 |
| `07_reduce_sum.mlir` | `15_reduce_sum.py` | 求和 | Σx | `reduction` | `tl.sum(x)` | LayerNorm |
| `08_reduce_max.mlir` | `16_reduce_max.py` | 最大值 | max(x) | `reduction+select` | `tl.max(x)` | Softmax 稳定 |
| `09_softmax_complete.mlir` | `17_softmax_complete.py` | Softmax | e^{x-max} | `reduce+broadcast+exp` | `tl.exp(x-tl.max(x))` | Attention |
| `10_clamp.mlir` | `18_clamp.py` | Clamp | clamp(-1,1) | `cmpf+select`×2 | `tl.clamp(x,-1,1)` | 梯度裁剪 |
| `11_layer_norm.mlir` | `19_layer_norm.py` | LayerNorm | (x-μ)/√(σ²+ε) | `subf+mulf` | `tl.mean+tl.var` | Transformer |

## 03_advanced (⭐⭐⭐ 复杂) — 10 个

| MLIR | Triton | 操作 | 公式 | MLIR 模式 | Triton 模式 | AI 角色 |
|------|--------|------|------|-----------|------------|---------|
| `01_matmul.mlir` | `20_matmul.py` | 矩阵乘 | C=A@B | `linalg.matmul` 1→74行 | `tl.dot(a,b)` | LLM 核心 |
| `10_matmul_4x4x4.mlir` | `20_matmul.py` | 矩阵乘(原始) | C=A@B | 同 `01_matmul.mlir` | 同 `20_matmul.py` | 4x4原始版 |
| `02_gemm_relu.mlir` | `21_gemm_relu.py` | GEMM+ReLU | ReLU(A@B) | 2 阶段 pipeline | `tl.max(tl.dot(),0)` | 融合 MLP |
| `03_depthwise_conv.mlir` | `26_depthwise_conv.py` | 深度卷积 | 逐通道 3×3 | `depthwise_conv named` | 手动窗口 | MobileNet |
| `04_conv2d.mlir` | `22_conv2d.py` | 二维卷积 | 输入×核 | `generic+reduction`×2 | 手动窗口+`tm.sum` | CNN 视觉 |
| `05_max_pool.mlir` | `23_max_pool.py` | 最大池化 | 2×2 取 max | `affine.for`×4 | `reshape+tl.max` | 下采样 |
| `06_avg_pool.mlir` | `24_avg_pool.py` | 平均池化 | 2×2 取 avg | `affine.for`×4 | `reshape+tl.sum/4` | 下采样 |
| `07_global_avg_pool.mlir` | `25_global_avg_pool.py` | 全局平均池化 | 全图 1 个平均值 | `affine.for`×2 | `tl.sum(x)/N` | 分类头 |
| `08_batch_norm_part1.mlir` | `27_batch_norm_part1.py` | BN 均值 | μ=Σx/N | `reduction+parallel` | `tl.sum(x)/N` | 训练稳定 |
| `09_batch_norm_part2.mlir` | `28_batch_norm_part2.py` | BN 标准化 | γ(x-μ)/√(σ²+ε)+β | 5×`ins`+`sqrt` | 5 行组合 | 归一化 |

## 降级膨胀对比

| 操作 | MLIR 输入 | LLVM 输出 | 膨胀率 | Triton |
|------|-----------|-----------|--------|--------|
| VecAdd (01_basic) | 3 行 | 38 行 | 12.7× | 1 行 |
| Tanh (01_basic) | 4 行 | 17 行 | 4.3× | 1 行 |
| ReduceMax (02_intermediate) | 5 行 | 47 行 | 9.4× | 1 行 |
| **MatMul (03_advanced)** | **1 行** | **74 行** | **74×** | **1 行** |
| Conv2D (03_advanced) | 6 行 | 85 行 | 14.2× | 4 行 |
| MaxPool (03_advanced) | 11 行 | 80 行 | 7.3× | 3 行 |
| BN Part2 (03_advanced) | 9 行 | 87 行 | 9.7× | 5 行 |

## 完整路径

| 组件 | 路径 |
|------|------|
| MLIR 文件 | `test-cases/mlir/{level}/{name}.mlir` |
| Triton 文件 | `test-cases/triton/{level}/{name}.py` |
| 双向映射 | `test-cases/triton/MAPPING.md` |
