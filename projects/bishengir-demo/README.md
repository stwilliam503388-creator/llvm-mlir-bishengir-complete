# bishengir-demo — 可运行 MLIR 降级流水线

用标准 `mlir-opt` 模拟 bishengir 三阶段降级（Linalg → HFusion → HIVM）。

---

## 测试用例（21 个）

覆盖 7 类神经网络常见算子。

### 逐元素运算（8 个）

| 文件 | 公式 | 对应 AI 场景 | LLVM 行数 |
|------|------|-------------|-----------|
| `vecadd_128.mlir` | `C = A + B` | 残差连接 | 38 |
| `relu_4x4.mlir` | `max(0, x)` | 全模型通用 | 42 |
| `leaky_relu_4.mlir` | `max(0.01x, x)` | GAN / 解决 ReLU 死亡 | 37 |
| `prelu_4x4.mlir` | `max(αx, x)` | 图像超分辨率 | 52 |
| `gelu_tanh_4.mlir` | `0.5x(1+tanh(x))` | BERT/GPT | 38 |
| `silu_4.mlir` | `x·σ(x)` | LLaMA 系列 | 23 |
| `sigmoid_4.mlir` | `1/(1+e^(-x))` | 二分类 / RNN 门控 | 33 |
| `hard_sigmoid_4.mlir` | `clamp(0.2x+0.5)` | MobileNet | 39 |

### 激活函数（5 个，包含在上表中，按用途独立列出）

| 文件 | 类型 | 计算量 | 是否可微 | 主要模型 |
|------|------|--------|---------|---------|
| `relu_4x4` | 分段线性 | 最低 | 分段可微 | CNN 全系 |
| `leaky_relu_4` | 分段线性 | 低 | 是 | GAN |
| `gelu_tanh_4` | 平滑 | 中 | 是 | BERT/GPT |
| `silu_4` | 平滑 | 中高 | 是 | LLaMA |
| `sigmoid_4` | S 形 | 高 | 是 | RNN |

### 归约操作（3 个）

| 文件 | 操作 | 用途 | LLVM 行数 |
|------|------|------|-----------|
| `reduce_sum_4x4.mlir` | 求和 | Layer Norm 分母 | 37 |
| `reduce_max_4x4.mlir` | 最大值 | Softmax 数值稳定 | 47 |
| `softmax_complete_4.mlir` | 减最大值 + 指数 | Attention 核心 | 44 |

### 规范化（2 个）

| 文件 | 操作 | 用途 | LLVM 行数 |
|------|------|------|-----------|
| `layer_norm_4x4.mlir` | 平方差 | Transformer 归一化 | 44 |
| `clamp_4x4.mlir` | 数值裁剪 | 梯度裁剪 | 53 |

### 矩阵运算（3 个）

| 文件 | 操作 | 用途 | LLVM 行数 |
|------|------|------|-----------|
| `matmul_4x4x4.mlir` | 矩阵乘法 | Linear/MLP 层 | **74** |
| `gemm_relu_4x4.mlir` | 矩阵乘 + ReLU | 融合 Linear+激活 | **77** |
| `depthwise_conv_4x4.mlir` | 深度卷积 | MobileNet | **113** |

### 卷积与池化（5 个）

| 文件 | 操作 | 实现方式 | LLVM 行数 |
|------|------|---------|-----------|
| `conv2d_4x4.mlir` | 二维卷积 4x4 (valid) | `linalg.generic` | 85 |
| `fill_4x4.mlir` | 张量填充 | `linalg.generic` | 39 |
| `max_pool_4x4.mlir` | 最大池化 2x2 (stride 2) | `affine.for` 手动 | 80 |
| `avg_pool_4x4.mlir` | 平均池化 2x2 (stride 2) | `affine.for` 手动 | 83 |
| `global_avg_pool_4x4.mlir` | 全局平均池化 | `affine.for` 手动 | 52 |

> pooling 用 `affine.for` 而非 `linalg.generic`，因为 linalg.generic 要求索引可逆，
> 而 stride>1 导致非可逆映射。详见 `LIMITATIONS.md`。

### 批归一化（2 个，需串联使用）

| 文件 | 步骤 | 用途 | LLVM 行数 |
|------|------|------|-----------|
| `batch_norm_4x4_part1.mlir` | reduce: Σx[i] → mean | 计算均值 | 48 |
| `batch_norm_4x4_part2.mlir` | normalize: γ·(x-μ)/√(σ²+ε)+β | 归一化 | 87 |

### 其他（3 个）

| 文件 | 操作 | 用途 | LLVM 行数 |
|------|------|------|-----------|
| `broadcast_4x4.mlir` | 标量广播 | Bias 加法 | 24 |
| `dropout_4x4.mlir` | 缩放（简化） | 防止过拟合 | 24 |
| `fused_128.mlir` | add + mul 融合 | 算子融合演示 | 59 |

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
