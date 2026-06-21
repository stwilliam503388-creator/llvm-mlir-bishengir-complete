# bishengir-demo — 可运行 MLIR 降级流水线

用标准 `mlir-opt` 模拟 bishengir 三阶段降级（Linalg → HFusion → HIVM）。

---

## 测试用例（28 个）— 按难度分级

> ⭐ = 入门（读完 Primer 就能看）
> ⭐⭐ = 进阶（需要 LLVM IR 基础）
> ⭐⭐⭐ = 复杂（需要 MLIR dialect / lowering 概念）

| 难度 | 数量 | 涉及概念 | 适合谁 |
|------|------|---------|--------|
| ⭐ 入门 | 8 | `linalg.generic` + parallel | 读完 Primer 即可 |
| ⭐⭐ 进阶 | 11 | `reduction` iterator、组合模式、条件分支 | 有 LLVM IR 基础 |
| ⭐⭐⭐ 复杂 | 9 | `linalg.matmul`、named op、`affine.for`、多步 pipeline | 理解 MLIR dialect 后 |

> ⭐ = 入门（读完 Primer 就能看）
> ⭐⭐ = 进阶（需要 LLVM IR 基础）
> ⭐⭐⭐ = 复杂（需要 MLIR dialect / lowering 概念）

---

### 第一梯队：入门 ⭐（8 个）

最基础的逐元素运算，只需理解 `linalg.generic` 的 "parallel" iterator 即可。

#### vecadd_128（向量加法）— 残差连接 / shortcut

| 维度 | 内容 |
|------|------|
| **功能** | `C[i] = A[i] + B[i]`，逐元素向量加法 |
| **AI 角色** | **残差连接 (Residual Connection/Shortcut)**：ResNet、Transformer 每层输出 + 输入直接相加，解决深层网络梯度消失。LLM 中每个 Attention/FFN 层后都有 `x + sublayer(x)` |
| **MLIR 模式** | `linalg.generic` + `arith.addf`，3 行 → 38 行 LLVM（12.7×）|
| **对应 bishengir** | `hfusion.elemwise_binary {fun = add}` |

#### relu_4x4（ReLU 激活）— 全模型通用激活函数

| 维度 | 内容 |
|------|------|
| **功能** | `y = max(0, x)`，负值截断为 0 |
| **AI 角色** | **ReLU (Rectified Linear Unit)**：CNN 全系标配，计算量最低的激活函数，GPU 友好。LLM 中 FFN 层用 ReLU 变体 (GELU/SwiGLU) 替代 |
| **MLIR 模式** | `arith.cmpf` + `arith.select`，条件分支 |
| **对应 bishengir** | `hfusion.elemwise_unary {fun = relu}` |

#### tanh_4（Tanh 激活）— RNN / LSTM

| 维度 | 内容 |
|------|------|
| **功能** | `y = tanh(x)`，S 形函数，输出范围 (-1, 1) |
| **AI 角色** | **Tanh (Hyperbolic Tangent)**：RNN/LSTM 的默认激活函数，用于控制信息流。LLM 中较少使用，但在某些门控机制中仍有出现 |
| **MLIR 模式** | `math.tanh` 内建函数调用 |
| **对应 bishengir** | `hfusion.elemwise_unary {fun = tanh}` |

#### softmax_4（Softmax 指数）— Attention 核心

| 维度 | 内容 |
|------|------|
| **功能** | `y = exp(x)`，指数运算（softmax 的前半部分）|
| **AI 角色** | **Softmax 指数**：Transformer 的 Attention 机制核心——计算 query 和 key 的匹配分数。完整 softmax = `exp(x - max(x)) / Σexp(x - max(x))`，本文件只演示 exp 部分 |
| **MLIR 模式** | `math.exp` 内建函数 |
| **对应 bishengir** | 逐元素 `math` 操作映射 |

#### broadcast_4x4（标量广播）— Bias 加法

| 维度 | 内容 |
|------|------|
| **功能** | `B[i][j] = A`，把标量复制到矩阵每个位置 |
| **AI 角色** | **Broadcasting (广播)**：神经网络的基础操作——卷积层的 bias、Batch Norm 的 γ/β、Attention 中的位置编码都需要 broadcast 后与特征图相加 |
| **MLIR 模式** | `affine_map<(i,j) -> ()>`，标量→矩阵 |
| **对应 bishengir** | `hfusion.broadcast` |

#### dropout_4x4（Dropout 训练）— 防止过拟合

| 维度 | 内容 |
|------|------|
| **功能** | `y = x * scale`，训练时按概率缩放（简化版，不含 mask）|
| **AI 角色** | **Dropout (随机丢弃)**：训练时随机忽略部分神经元，防止过拟合。Transformer 早期使用（BERT），现代 LLM (GPT-4/LLaMA) 趋向于不用 |
| **MLIR 模式** | `arith.mulf` 逐元素乘法 |
| **对应 bishengir** | `hfusion.elemwise_binary {fun = mul}` |

#### fill_4x4（张量填充）— 初始化缓冲区

| 维度 | 内容 |
|------|------|
| **功能** | 用常数值填充整个张量 |
| **AI 角色** | **初始化 (Initialization)**：在卷积/矩阵乘前初始化输出缓冲区。几乎每个模型的第一层和中间层都会用到 |
| **MLIR 模式** | `linalg.generic` + `yield %cst` |
| **对应 bishengir** | `linalg.fill`（Homebrew 未编译，用 generic 替代）|

#### fused_128（add + mul 融合）— 算子融合概念

| 维度 | 内容 |
|------|------|
| **功能** | 连续两个 `linalg.generic`：先加后乘 |
| **AI 角色** | **算子融合 (Kernel Fusion)**：编译器核心优化——将连续两个 kernel 合并为一个，减少内存读写。深度学习编译器 (TVM/XLA) 的核心能力 |
| **MLIR 模式** | 连续 `linalg.generic` 两次 |
| **对应 bishengir** | HFusion 的"算子融合"概念演示 |

---

### 第二梯队：进阶 ⭐⭐（11 个）

需要理解 `reduction` iterator、`arith.cmpf` 条件分支、以及连续多步运算。

#### sigmoid_4（Sigmoid 激活）— 二分类 / RNN 门控

| 维度 | 内容 |
|------|------|
| **功能** | `σ(x) = 1 / (1 + e^{-x})`，输出范围 (0, 1) |
| **AI 角色** | **Sigmoid (Logistic 函数)**：二分类输出层、RNN 的 forget/input/output 门控。LLaMA 等现代 LLM 也用它做门控激活 (SwiGLU) |
| **MLIR 模式** | `arith.negf` + `math.exp` + `arith.addf` + `arith.divf`，4 步组合 |
| **难度提示** | 需要理解 `arith.negf` 和 `math.exp` 的配合 |

#### silu_4（SiLU / Swish 激活）— LLaMA 系列

| 维度 | 内容 |
|------|------|
| **功能** | `silu(x) = x * σ(x)`，Sigmoid 门控的输入 |
| **AI 角色** | **SiLU (Sigmoid Linear Unit)**：LLaMA 2/3、Mistral、Gemma 等现代 LLM 的 FFN 层使用 SwiGLU (SiLU 的变体)。比 ReLU 平滑，比 GELU 计算量低 |
| **MLIR 模式** | `sigmoid` + `arith.mulf`，5 步组合 |
| **对应 bishengir** | 组合模式→可融合为单个 hfusion |

#### leaky_relu_4（Leaky ReLU）— GAN

| 维度 | 内容 |
|------|------|
| **功能** | `y = x if x > 0 else 0.01*x`，负数侧有微小斜率 |
| **AI 角色** | **Leaky ReLU (带泄漏的线性整流)**：解决 ReLU 死亡问题（负数区梯度为零）。GAN (生成对抗网络) 标配，部分传统 CNN 也使用 |
| **MLIR 模式** | `arith.cmpf` + `arith.mulf` + `arith.select` |
| **对应 bishengir** | 条件分支模式 |

#### gelu_tanh_4（GELU 近似）— BERT / GPT

| 维度 | 内容 |
|------|------|
| **功能** | `gelu(x) ≈ 0.5 * x * (1 + tanh(x))` |
| **AI 角色** | **GELU (Gaussian Error Linear Unit)**：BERT/GPT-2/GPT-3 的 FFN 激活函数。比 ReLU 平滑，性能更好。GPT-4 和 LLaMA 改用 SwiGLU |
| **MLIR 模式** | `math.tanh` + `arith.addf` + `arith.mulf`，4 步组合 |
| **对应 bishengir** | 组合模式可融合 |

#### hard_sigmoid_4（Hard Sigmoid）— MobileNet

| 维度 | 内容 |
|------|------|
| **功能** | `hard_sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)`，sigmoid 的线性近似 |
| **AI 角色** | **Hard Sigmoid (硬 Sigmoid)**：MobileNetV3 等轻量化模型使用，计算量比 sigmoid 小 3×。适合移动端部署 |
| **MLIR 模式** | `arith.maximumf` + `arith.minimumf`，数值裁剪 |
| **对应 bishengir** | 分段线性函数映射 |

#### prelu_4x4（PReLU）— 图像超分辨率

| 维度 | 内容 |
|------|------|
| **功能** | `y = x if x > 0 else α*x`，α 是可训练参数 |
| **AI 角色** | **PReLU (Parametric ReLU, 参数化线性整流)**：Leaky ReLU 的扩展，α 通过梯度下降学习。ESPCN/SRGAN 等超分辨率模型常用 |
| **MLIR 模式** | `arith.mulf` + `arith.cmpf` + `arith.select` |

#### reduce_sum_4x4（求和归约）— Layer Norm 分母

| 维度 | 内容 |
|------|------|
| **功能** | `sum = ΣᵢΣⱼ x[i][j]`，矩阵所有元素求和 |
| **AI 角色** | **Sum Reduction (求和归约)**：Layer Norm 需要计算 `mean = Σx / N` 和 `variance = Σ(x - mean)² / N`，求和是第一步 |
| **MLIR 模式** | **`reduction` iterator**，多维→标量 |
| **对应 bishengir** | `hfusion.reduce {fun = add, axes = [0, 1]}` |
| **难度提示** | 这是第一个 `reduction` 用法的例子，理解后其他 reduction 都类似 |

#### reduce_max_4x4（最大值归约）— Softmax 数值稳定

| 维度 | 内容 |
|------|------|
| **功能** | `max = max(all elements)`，求矩阵最大值 |
| **AI 角色** | **Max Reduction (最大值归约)**：softmax 数值稳定性关键——`softmax(x) = exp(x - max(x)) / Σexp(x - max(x))`，先减去最大值防止 exp 溢出 |
| **MLIR 模式** | `reduction` + `arith.cmpf` + `arith.select` |

#### softmax_complete_4（完整 Softmax 第一步）— Attention 核心

| 维度 | 内容 |
|------|------|
| **功能** | `y = exp(x - max(x))`，减最大值后取指数 |
| **AI 角色** | **数值稳定 Softmax**：Attention 机制的完整第一步。Transformer 中的 `softmax(QK^T / √d)` 依赖于数值稳定的 exp |
| **MLIR 模式** | `memref.alloc` + `reduction` + `memref.store` |
| **难度提示** | 需要理解 memref 的手动分配和写入 |

#### clamp_4x4（数值裁剪）— 梯度裁剪

| 维度 | 内容 |
|------|------|
| **功能** | `y = clamp(x, min, max)`，限制值在 [min, max] 区间 |
| **AI 角色** | **Gradient Clipping (梯度裁剪)**：训练时限制梯度范围，防止梯度爆炸。LLM 训练必用。推理时可用于激活值截断（量化友好） |
| **MLIR 模式** | 两次 `arith.cmpf` + `arith.select` |
| **对应 bishengir** | 分段线性函数映射 |

---

### 第三梯队：复杂 ⭐⭐⭐（9 个）

需要理解 `linalg.matmul` named op、多步 pipeline、手动循环 (affine.for)、以及完整的 BN/LayerNorm 计算链。

#### matmul_4x4x4（矩阵乘法）— Linear / MLP 层

| 维度 | 内容 |
|------|------|
| **功能** | `C = A @ B` (矩阵乘法)，4×4 × 4×4 → 4×4 |
| **AI 角色** | **Linear Layer (全连接层 / 线性层)**：`y = x @ W^T + b`。LLM 中 Attention 的 Q/K/V 投影、FFN 的 up/down projection 全部是 matmul。**这是 AI 模型最核心的算子，占算力 60-80%** |
| **MLIR 模式** | `linalg.matmul` named op，1 行 → **74 行 LLVM**（74× 膨胀）|
| **对应 bishengir** | `hfusion.cube_matmul` → `hivm.mmul`（1 行，硬件指令）|
| **难度提示** | 74× 膨胀不是问题——bishengir 保持 1 行 NPU 指令。膨胀展示了"不保留语义会怎样" |

#### gemm_relu_4x4（矩阵乘 + ReLU 融合）— MLP 标准模式

| 维度 | 内容 |
|------|------|
| **功能** | `y = ReLU(x @ W)`，先矩阵乘后激活 |
| **AI 角色** | **算子融合 (GEMM + Activation)**：MLP 层的标准模式——`x @ W1 + b → ReLU → x @ W2 + b`。编译器可以将 matmul + relu 融合为单个 kernel，减少一次中間 buffer 读写 |
| **MLIR 模式** | `linalg.matmul` + `linalg.generic` 两阶段 pipeline |
| **对应 bishengir** | 融合优化 |
| **难度提示** | 需要理解两阶段 lowering 的配合 |

#### depthwise_conv_4x4（深度卷积）— MobileNet

| 维度 | 内容 |
|------|------|
| **功能** | 逐通道 3×3 卷积，输入 1×4×4×1，输出 1×4×4×1 |
| **AI 角色** | **Depthwise Convolution (深度可分离卷积)**：MobileNet/EfficientNet 的核心算子，计算量是标准卷积的 1/C (C 为通道数)。结合 pointwise conv 组成 depthwise separable conv |
| **MLIR 模式** | `linalg.depthwise_conv_2d_nhwc_hwcm` named op，3 行 → 113 行 LLVM |
| **对应 bishengir** | named op 映射 |
| **难度提示** | 113 行 LLVM = 稠密的卷积展开，最复杂的单独 op |

#### conv2d_4x4（二维卷积）— 标准卷积层

| 维度 | 内容 |
|------|------|
| **功能** | 2D valid 卷积：输入 4×4，kernel 3×3，输出 2×2 |
| **AI 角色** | **Convolution (卷积)**：CNN 的绝对核心——ResNet/VGG/YOLO/UNet 等视觉模型全部依赖卷积。LLM 中虽然不直接用，但多模态模型 (GPT-4V) 的视觉编码器仍用卷积 |
| **MLIR 模式** | `linalg.generic` + `reduction` × 2，6 行 → 85 行 LLVM |
| **对应 bishengir** | 可被 bishengir 模式匹配优化 |
| **难度提示** | 需要理解 `reduction` iterator 与 affine_map 的配合 |

#### max_pool_4x4（最大池化）— 下采样

| 维度 | 内容 |
|------|------|
| **功能** | 2×2 窗口内取最大值，stride=2，4×4 → 2×2 |
| **AI 角色** | **Max Pooling (最大池化)**：CNN 的下采样层——保留最强激活值，丢弃位置信息。经典 CNN (LeNet/AlexNet/VGG) 标配，现代模型趋向于用 stride=2 的卷积替代 |
| **MLIR 模式** | `affine.for` × 4 + `arith.cmpf` + `select` |
| **对应 bishengir** | `linalg.pooling_nhwc_max`（Homebrew 不可用，用 affine 替代）|
| **难度提示** | 需要理解手动循环的 affine.for 语法 |

#### avg_pool_4x4（平均池化）— 下采样

| 维度 | 内容 |
|------|------|
| **功能** | 2×2 窗口内取平均值，stride=2，4×4 → 2×2 |
| **AI 角色** | **Average Pooling (平均池化)**：比 max pooling 更平滑的下采样。ResNet 中使用 `avg_pool` 做分类头前的下采样 |
| **MLIR 模式** | `affine.for` × 4 + 累加 + 除法 |
| **对应 bishengir** | `linalg.pooling_nchw_sum` + 除法 |

#### global_avg_pool_4x4（全局平均池化）— 分类头

| 维度 | 内容 |
|------|------|
| **功能** | 对整个特征图求平均，4×4 → 1×1 |
| **AI 角色** | **Global Average Pooling (全局平均池化, GAP)**：ResNet/MobileNet/GoogleNet 的分类头——代替全连接层，将特征图压缩为类别置信度。参数量为 0，天然防止过拟合 |
| **MLIR 模式** | `affine.for` × 2 + 累加 + 平均因子 |
| **难度提示** | 2 层循环比 4 层循环简单，但需要理解 `memref<f32>` 标量存储 |

#### batch_norm_4x4_part1 + part2（批归一化）— 训练稳定性

| 维度 | 内容 |
|------|------|
| **功能** | Part1: 每个通道求均值 `mean[j] = Σᵢ x[i][j] / N`，Part2: `y = γ × (x - μ) / √(σ² + ε) + β` |
| **AI 角色** | **Batch Normalization (批归一化, BN)**：CNN 训练的核心技巧——稳定训练、允许更高学习率。ResNet 中每层卷积后都有 BN。LLM 中已被 Layer Norm 替代，但视觉模型仍用 BN |
| **MLIR 模式** | Part1: `reduction` + `parallel` 混合 iterator；Part2: 5 个 ins 操作数的 `linalg.generic` |
| **对应 bishengir** | 需拆解为 reduce + broadcast + elemwise 组合 |
| **难度提示** | ⭐⭐⭐ 最复杂的用例——需要理解多步 pipeline、reduction 与 broadcast 的配合 |

#### layer_norm_4x4（层归一化）— Transformer

| 维度 | 内容 |
|------|------|
| **功能** | `y = (x - mean) / sqrt(var + eps) * γ + β`，平方差部分 |
| **AI 角色** | **Layer Normalization (层归一化, LN)**：Transformer 的标配归一化——每个 token 自己做归一化，不依赖 batch 内其他 token。GPT/BERT/LLaMA 每层都有 LN。**比 BN 更适合变长序列** |
| **MLIR 模式** | `linalg.generic` + `arith.subf` + `arith.mulf` |
| **对应 bishengir** | 组合模式 |
| **难度提示** | 概念上相对简单（逐元素），但与 reduce 配合才能完成完整的 LN |

---

## 学习路线建议

```
初学者 → 先看 ⭐ 8 个 → 建立 linalg.generic 直觉
    ↓
有一点点基础 → 看 ⭐⭐ 11 个 → 理解 reduction / broadcast / 组合模式
    ↓
掌握 MLIR 核心概念 → 看 ⭐⭐⭐ 9 个 → matmul / conv / 多步 pipeline
    ↓
理解 bishengir 降级 → 再跑一遍 variants/compare.sh → 观察 28 个用例的膨胀率
```

---

## 快速开始

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# 单个用例
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir

# 完整降级到 LLVM
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  test-cases/vecadd_128.mlir

# 批量运行
bash run-demo.sh
```

---

## 矩阵乘法优化方案对比

matmul 的 74× 膨胀源于三重循环完全展开为标量。
`variants/compare.sh` 直接对比 4 种优化策略：

| Variant | 策略 | LLVM 行数 | vs 基准 | 原理 |
|---------|------|-----------|---------|------|
| **V0** | 无优化 (基准) | 74 行 | - | 三重循环完全展开 |
| **V1** | 循环分块 (tile=2x2x1) | 76 行 | +2 行 | 增加 tile 循环层，改善 cache |
| **V2** | 向量化 (tile+vectorize) | 77 行 | +3 行 | SIMD 指令，减少指令数 |
| **V3** | **硬件映射 (模拟 mmul)** | **5 行** | **-69 行 (-93%)** | func.call 保留语义，不展开 |

### V3 的核心思路 — bishengir 实际采用的方案

```text
标准 MLIR 路径 (V0):
  linalg.matmul → affine.for×3 → scf.for+arith → llvm.load/add/mul/store  (74行)

bishengir 路径 (≈V3):
  linalg.matmul → hfusion.cube_matmul (1行) → hivm.mmul (1行)
                                                 ↑
                                           Ascend NPU Cube 单元
                                           硬件直接执行矩阵乘
```

**关键**: 高级操作**保持高级语义**（不展开到标量），直接映射到硬件指令。

### 环境限制

详见 `LIMITATIONS.md`。

**一句话总结**: Homebrew LLVM 22 未编译 Linalg named ops（conv/pooling/fill 等 named 版本），
但全部可通过 `linalg.generic` 替代（pooling 除外——需 bishengir 自编译版本）。
bishengir-opt 自编译时包含这些 named op，功能不受影响。

```bash
bash variants/compare.sh
```

---

## bishengir ↔ 标准 MLIR 对照

```text
bishengir:                      标准 MLIR (本 demo):
────────────────────             ────────────────────
linalg.generic                  linalg.generic
    ↓ -convert-linalg-to-hfusion    ↓ --convert-linalg-to-affine-loops
hfusion.elemwise_binary         affine.for + arith.addf
    ↓ -convert-hfusion-to-hivm     ↓ --lower-affine --scf-to-cf --func-to-llvm
hivm.load/vadd/store            llvm.load + llvm.add + llvm.store
```
