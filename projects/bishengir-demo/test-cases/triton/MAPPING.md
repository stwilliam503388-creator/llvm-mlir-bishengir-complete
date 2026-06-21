1|# Triton ↔ MLIR 映射对照表
2|
3|每个 Triton 文件对应一个 MLIR 测试用例。MLIR 文件在 `test-cases/01_basic/` / `mlir/02_intermediate/` / `mlir/03_advanced/` 下。
4|
5|> 💡 逆向映射见 `../mlir/MAPPING.md`（从 MLIR 查 Triton）。

> ⭐ = 入门 | ⭐⭐ = 进阶 | ⭐⭐⭐ = 复杂
6|
7|---
8|
9|## 01_basic (⭐ 入门)
10|
11|| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
12||--------|------|------|------|-------------|------------|--------------|
13|| `01_vecadd.py` | `mlir/01_basic/01_vecadd.mlir` | 向量加法 | C[i]=A[i]+B[i] | `linalg.generic` + `arith.addf` | `a + b` | 残差连接, Transformer 每层 |
14|| `02_relu.py` | `mlir/01_basic/02_relu.mlir` | ReLU 激活 | max(0,x) | `arith.cmpf` + `select` | `tl.maximum(x, 0)` | CNN 标配激活 |
15|| `03_tanh.py` | `mlir/01_basic/03_tanh.mlir` | Tanh 激活 | (-1,1) | `math.tanh` | `tl.tanh(x)` | RNN 门控 |
16|| `04_softmax_exp.py` | `mlir/01_basic/04_softmax_exp.mlir` | Softmax 指数 | e^x | `math.exp` | `tl.exp(x)` | Attention 核心 |
17|| `05_broadcast.py` | `mlir/01_basic/05_broadcast.mlir` | 标量广播 | B[i]=A | `affine_map<()->(i)>` | 自动广播 | bias/Norm 参数 |
18|| `06_dropout.py` | `mlir/01_basic/06_dropout.mlir` | Dropout | x*scale | `arith.mulf` | `x * scale` | 训练正则化 |
19|| `07_fill.py` | `mlir/01_basic/07_fill.mlir` | 常量填充 | A[i]=c | `linalg.generic` + yield | `store(val)` | 缓冲区初始化 |
20|| `08_fused.py` | `mlir/01_basic/08_fused.mlir` | 算子融合 | C=A+B; D=C*A | 连续 2 次 `linalg.generic` | 1 个 kernel 内做完 | 编译器融合优化 |
21|
22|## 02_intermediate (⭐⭐ 进阶)
23|
24|| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
25||--------|------|------|------|-------------|------------|--------------|
26|| `09_sigmoid.py` | `mlir/02_intermediate/01_sigmoid.mlir` | Sigmoid | 1/(1+e^{-x}) | 4 步: negf+exp+addf+divf | `tl.sigmoid(x)` | 二分类, RNN 门控 |
27|| `10_silu.py` | `mlir/02_intermediate/02_silu.mlir` | SiLU / Swish | x*sigmoid(x) | 5 步组合 | `x * tl.sigmoid(x)` | LLaMA FFN 层 |
28|| `11_leaky_relu.py` | `mlir/02_intermediate/03_leaky_relu.mlir` | LeakyReLU | max(x, 0.01x) | `cmpf` + `mulf` + `select` | `tl.where(x>0,x,0.01*x)` | GAN 标配 |
29|| `12_gelu_tanh.py` | `mlir/02_intermediate/04_gelu_tanh.mlir` | GELU (tanh) | 0.5x(1+tanh(x)) | `tanh`+`addf`+`mulf` x2 | `0.5*x*(1+tl.tanh(x))` | BERT/GPT-3 FFN |
30|| `13_hard_sigmoid.py` | `mlir/02_intermediate/05_hard_sigmoid.mlir` | Hard Sigmoid | clamp(0.2x+0.5,0,1) | `maximumf`+`minimumf` | `tl.clamp(0.2*x+0.5,0,1)` | MobileNetV3 |
31|| `14_prelu.py` | `mlir/02_intermediate/06_prelu.mlir` | PReLU | x>0?x:alpha*x | `mulf`+`cmpf`+`select` | `tl.where(x>0,x,alpha*x)` | 超分辨率 |
32|| `15_reduce_sum.py` | `mlir/02_intermediate/07_reduce_sum.mlir` | 归约求和 | sum(x_i) | `reduction` iterator | `tl.sum(x)` | LayerNorm, Softmax |
33|| `16_reduce_max.py` | `mlir/02_intermediate/08_reduce_max.mlir` | 归约最大值 | max(x_i) | `reduction`+`cmpf`+`select` | `tl.max(x)` | Softmax 数值稳定 |
34|| `17_softmax_complete.py` | `mlir/02_intermediate/09_softmax_complete.mlir` | Softmax 稳定 | e^{x-max(x)} | `reduce`+`broadcast`+`exp` | `tl.exp(x - tl.max(x))` | Attention |
35|| `18_clamp.py` | `mlir/02_intermediate/10_clamp.mlir` | 数值裁剪 | clamp(x,-1,1) | `cmpf`+`select` x2 | `tl.clamp(x, -1, 1)` | 梯度裁剪, 量化 |
36|| `19_layer_norm.py` | `mlir/02_intermediate/11_layer_norm.mlir` | LayerNorm | (x-μ)/√(σ²+ε) | `subf`+`mulf`+`sqrt` | `tl.mean`+`tl.var` | Transformer 每层 |
37|
38|## 03_advanced (⭐⭐⭐ 复杂)
39|
40|| Triton | MLIR | 操作 | 公式 | MLIR 核心模式 | Triton 模式 | 在 AI 中的角色 |
41||--------|------|------|------|-------------|------------|--------------|
42|| `20_matmul.py` | `mlir/03_advanced/01_matmul.mlir` | 矩阵乘法 | C=A@B | `linalg.matmul`, 1→74行 | `tl.dot(a, b)` | LLM 算力 60-80% |
43|| `21_gemm_relu.py` | `mlir/03_advanced/02_gemm_relu.mlir` | GEMM+ReLU | ReLU(A@B) | 2 阶段 pipeline | `tl.maximum(tl.dot(a,b),0)` | 融合 MLP 层 |
44|| `22_conv2d.py` | `mlir/03_advanced/04_conv2d.mlir` | 二维卷积 | sum(输入×核) | `generic`+`reduction` x2 | 手动窗口+`tl.sum` | CNN 视觉核心 |
45|| `23_max_pool.py` | `mlir/03_advanced/05_max_pool.mlir` | 最大池化 | 2×2窗口取 max | `affine.for` x4 + `select` | `tl.reshape`+`tl.max` | CNN 下采样 |
46|| `24_avg_pool.py` | `mlir/03_advanced/06_avg_pool.mlir` | 平均池化 | 2×2窗口取 avg | `affine.for` x4 + `addf`+`divf` | `tl.reshape`+`tl.sum`/4 | ResNet 采样 |
47|| `25_global_avg_pool.py` | `mlir/03_advanced/07_global_avg_pool.mlir` | 全局平均池化 | 全图 1 个平均值 | `affine.for` x2 + 累加 | `tl.sum(x)/N` | 分类头前 |
48|| `26_depthwise_conv.py` | `mlir/03_advanced/03_depthwise_conv.mlir` | 深度卷积 | 每通道独立卷积 | `depthwise_conv_2d` named op | 同 conv2d 模式 | MobileNet |
49|| `27_batch_norm_part1.py` | `mlir/03_advanced/08_batch_norm_part1.mlir` | BN 均值 | μ=Σx/N | `reduction`+`parallel` | `tl.sum(x)/N` | 训练稳定化 |
50|| `28_batch_norm_part2.py` | `mlir/03_advanced/09_batch_norm_part2.mlir` | BN 标准化 | γ(x-μ)/√(σ²+ε)+β | 5个 `ins` + `sqrt` | `gamma*(x-mu)/sqrt(var+eps)+beta` | 归一化通用模式 |
51|
52|## 关键发现
53|
54|### 1. 抽象层级差异
55|
56|MLIR 多个步骤的操作, Triton 一行搞定:
57|
58|| 操作 | MLIR 步骤数 | Triton 行数 | 原因 |
59||------|-----------|------------|------|
60|| Sigmoid | 4 (negf→exp→addf→divf) | 1 (`tl.sigmoid`) | Triton 封装为内建 |
61|| GELU | 4 (tanh→addf→mulf×2) | 1 (`0.5*x*(1+tl.tanh(x)`) | 纯数学组合 |
62|| MaxPool | 11 (4层循环+load+cmpf+select+store) | 3 (`reshape`+`tl.max`+`store`) | Triton 自动向量化 |
63|| MatMul | 74行展开的 LLVM IR | 1 (`tl.dot`) | Tensor core 硬件 |
64|
65|### 2. Triton 无法表达的操作
66|
67|| MLIR 概念 | Triton 等效 | 说明 |
68||-----------|------------|------|
69|| `memref.alloc`/`dealloc` | 无需 | Python/torch 管理内存 |
70|| `affine.for` 手动循环 | `tl.arange` + reshape | Triton 用向量化替代标量循环 |
71|| `linalg.generic` 通用模式 | 直接写运算 | Triton 用 Python 运算符 |
72|| `reduction` 归约迭代器 | `tl.sum`/`tl.max`/`tl.argmin` | 一维归约函数 |
73|
74|### 3. bishengir 特殊映射
75|
76|| bishengir 方言 | Triton 等效 | 说明 |
77||----------------|------------|------|
78|| `hfusion.elemwise_binary` | `a + b` / `a * b` | 逐元素操作 |
79|| `hfusion.cube_matmul` | `tl.dot()` | 矩阵乘 (NPU cube = GPU tensor core) |
80|| `hivm.vadd` | `a + b` | 向量指令 |
81|| `hivm.mmul` | `tl.dot()` | 矩阵指令 |
82|
83|## 使用方式
84|
85|```bash
86|# 在 NVIDIA GPU 上运行
87|cd projects/bishengir-demo/
88|python test-cases/triton/20_matmul.py   # 跑矩阵乘
89|python test-cases/triton/10_silu.py     # 跑SiLU
90|
91|# 对照 MLIR 理解降低过程
92|# 1. 读 Triton 代码理解高层语义
93|# 2. 读 MLIR 代码理解编译器中间表示
94|# 3. 用 mlir-opt 降级看到 LLVM IR
95|```
96|