# Triton ↔ MLIR 映射对照表

每个 Triton 文件对应一个 MLIR 测试用例。MLIR 文件在 `test-cases/01_basic/` / `02_intermediate/` / `03_advanced/` 下。

> ⭐ = 入门 | ⭐⭐ = 进阶 | ⭐⭐⭐ = 复杂

---

## 01_basic (⭐ 入门)

| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
|--------|------|------|------|-------------|------------|--------------|
| `01_vecadd.py` | `01_basic/01_vecadd.mlir` | 向量加法 | C[i]=A[i]+B[i] | `linalg.generic` + `arith.addf` | `a + b` | 残差连接, Transformer 每层 |
| `02_relu.py` | `01_basic/02_relu.mlir` | ReLU 激活 | max(0,x) | `arith.cmpf` + `select` | `tl.maximum(x, 0)` | CNN 标配激活 |
| `03_tanh.py` | `01_basic/03_tanh.mlir` | Tanh 激活 | (-1,1) | `math.tanh` | `tl.tanh(x)` | RNN 门控 |
| `04_softmax_exp.py` | `01_basic/04_softmax_exp.mlir` | Softmax 指数 | e^x | `math.exp` | `tl.exp(x)` | Attention 核心 |
| `05_broadcast.py` | `01_basic/05_broadcast.mlir` | 标量广播 | B[i]=A | `affine_map<()->(i)>` | 自动广播 | bias/Norm 参数 |
| `06_dropout.py` | `01_basic/06_dropout.mlir` | Dropout | x*scale | `arith.mulf` | `x * scale` | 训练正则化 |
| `07_fill.py` | `01_basic/07_fill.mlir` | 常量填充 | A[i]=c | `linalg.generic` + yield | `store(val)` | 缓冲区初始化 |
| `08_fused.py` | `01_basic/08_fused.mlir` | 算子融合 | C=A+B; D=C*A | 连续 2 次 `linalg.generic` | 1 个 kernel 内做完 | 编译器融合优化 |

## 02_intermediate (⭐⭐ 进阶)

| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
|--------|------|------|------|-------------|------------|--------------|
| `09_sigmoid.py` | `02_intermediate/01_sigmoid.mlir` | Sigmoid | 1/(1+e^{-x}) | 4 步: negf+exp+addf+divf | `tl.sigmoid(x)` | 二分类, RNN 门控 |
| `10_silu.py` | `02_intermediate/02_silu.mlir` | SiLU / Swish | x*sigmoid(x) | 5 步组合 | `x * tl.sigmoid(x)` | LLaMA FFN 层 |
| `11_leaky_relu.py` | `02_intermediate/03_leaky_relu.mlir` | LeakyReLU | max(x, 0.01x) | `cmpf` + `mulf` + `select` | `tl.where(x>0,x,0.01*x)` | GAN 标配 |
| `12_gelu_tanh.py` | `02_intermediate/04_gelu_tanh.mlir` | GELU (tanh) | 0.5x(1+tanh(x)) | `tanh`+`addf`+`mulf` x2 | `0.5*x*(1+tl.tanh(x))` | BERT/GPT-3 FFN |
| `13_hard_sigmoid.py` | `02_intermediate/05_hard_sigmoid.mlir` | Hard Sigmoid | clamp(0.2x+0.5,0,1) | `maximumf`+`minimumf` | `tl.clamp(0.2*x+0.5,0,1)` | MobileNetV3 |
| `14_prelu.py` | `02_intermediate/06_prelu.mlir` | PReLU | x>0?x:alpha*x | `mulf`+`cmpf`+`select` | `tl.where(x>0,x,alpha*x)` | 超分辨率 |
| `15_reduce_sum.py` | `02_intermediate/07_reduce_sum.mlir` | 归约求和 | sum(x_i) | `reduction` iterator | `tl.sum(x)` | LayerNorm, Softmax |
| `16_reduce_max.py` | `02_intermediate/08_reduce_max.mlir` | 归约最大值 | max(x_i) | `reduction`+`cmpf`+`select` | `tl.max(x)` | Softmax 数值稳定 |
| `17_softmax_complete.py` | `02_intermediate/09_softmax_complete.mlir` | Softmax 稳定 | e^{x-max(x)} | `reduce`+`broadcast`+`exp` | `tl.exp(x - tl.max(x))` | Attention |
| `18_clamp.py` | `02_intermediate/10_clamp.mlir` | 数值裁剪 | clamp(x,-1,1) | `cmpf`+`select` x2 | `tl.clamp(x, -1, 1)` | 梯度裁剪, 量化 |
| `19_layer_norm.py` | `02_intermediate/11_layer_norm.mlir` | LayerNorm | (x-μ)/√(σ²+ε) | `subf`+`mulf`+`sqrt` | `tl.mean`+`tl.var` | Transformer 每层 |

## 03_advanced (⭐⭐⭐ 复杂)

| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
|--------|------|------|------|-------------|------------|--------------|
| `20_matmul.py` | `03_advanced/01_matmul.mlir` | 矩阵乘法 | C=A@B | `linalg.matmul`, 1→74行 | `tl.dot(a, b)` | LLM 算力 60-80% |
| `21_gemm_relu.py` | `03_advanced/02_gemm_relu.mlir` | GEMM+ReLU | ReLU(A@B) | 2 阶段 pipeline | `tl.maximum(tl.dot(a,b),0)` | 融合 MLP 层 |
| `22_conv2d.py` | `03_advanced/04_conv2d.mlir` | 二维卷积 | sum(输入×核) | `generic`+`reduction` x2 | 手动窗口+`tl.sum` | CNN 视觉核心 |
| `23_max_pool.py` | `03_advanced/05_max_pool.mlir` | 最大池化 | 2×2窗口取 max | `affine.for` x4 + `select` | `tl.reshape`+`tl.max` | CNN 下采样 |
| `24_avg_pool.py` | `03_advanced/06_avg_pool.mlir` | 平均池化 | 2×2窗口取 avg | `affine.for` x4 + `addf`+`divf` | `tl.reshape`+`tl.sum`/4 | ResNet 采样 |
| `25_global_avg_pool.py` | `03_advanced/07_global_avg_pool.mlir` | 全局平均池化 | 全图 1 个平均值 | `affine.for` x2 + 累加 | `tl.sum(x)/N` | 分类头前 |
| `26_depthwise_conv.py` | `03_advanced/03_depthwise_conv.mlir` | 深度卷积 | 每通道独立卷积 | `depthwise_conv_2d` named op | 同 conv2d 模式 | MobileNet |
| `27_batch_norm_part1.py` | `03_advanced/08_batch_norm_part1.mlir` | BN 均值 | μ=Σx/N | `reduction`+`parallel` | `tl.sum(x)/N` | 训练稳定化 |
| `28_batch_norm_part2.py` | `03_advanced/09_batch_norm_part2.mlir` | BN 标准化 | γ(x-μ)/√(σ²+ε)+β | 5个 `ins` + `sqrt` | `gamma*(x-mu)/sqrt(var+eps)+beta` | 归一化通用模式 |

## 关键发现

### 1. 抽象层级差异

MLIR 多个步骤的操作, Triton 一行搞定:

| 操作 | MLIR 步骤数 | Triton 行数 | 原因 |
|------|-----------|------------|------|
| Sigmoid | 4 (negf→exp→addf→divf) | 1 (`tl.sigmoid`) | Triton 封装为内建 |
| GELU | 4 (tanh→addf→mulf×2) | 1 (`0.5*x*(1+tl.tanh(x)`) | 纯数学组合 |
| MaxPool | 11 (4层循环+load+cmpf+select+store) | 3 (`reshape`+`tl.max`+`store`) | Triton 自动向量化 |
| MatMul | 74行展开的 LLVM IR | 1 (`tl.dot`) | Tensor core 硬件 |

### 2. Triton 无法表达的操作

| MLIR 概念 | Triton 等效 | 说明 |
|-----------|------------|------|
| `memref.alloc`/`dealloc` | 无需 | Python/torch 管理内存 |
| `affine.for` 手动循环 | `tl.arange` + reshape | Triton 用向量化替代标量循环 |
| `linalg.generic` 通用模式 | 直接写运算 | Triton 用 Python 运算符 |
| `reduction` 归约迭代器 | `tl.sum`/`tl.max`/`tl.argmin` | 一维归约函数 |

### 3. bishengir 特殊映射

| bishengir 方言 | Triton 等效 | 说明 |
|----------------|------------|------|
| `hfusion.elemwise_binary` | `a + b` / `a * b` | 逐元素操作 |
| `hfusion.cube_matmul` | `tl.dot()` | 矩阵乘 (NPU cube = GPU tensor core) |
| `hivm.vadd` | `a + b` | 向量指令 |
| `hivm.mmul` | `tl.dot()` | 矩阵指令 |

## 使用方式

```bash
# 在 NVIDIA GPU 上运行
cd projects/bishengir-demo/
python test-cases/triton/20_matmul.py   # 跑矩阵乘
python test-cases/triton/10_silu.py     # 跑SiLU

# 对照 MLIR 理解降低过程
# 1. 读 Triton 代码理解高层语义
# 2. 读 MLIR 代码理解编译器中间表示
# 3. 用 mlir-opt 降级看到 LLVM IR
```
