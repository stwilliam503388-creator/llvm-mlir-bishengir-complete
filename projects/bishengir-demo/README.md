# bishengir-demo — 可运行 MLIR 降级流水线

用标准 `mlir-opt` 模拟 bishengir 三阶段降级（Linalg → HFusion → HIVM）。

---

## 测试用例（10 个）

### 逐元素操作（5 个）

| 文件 | 操作 | MLIR 核心模式 | 对应深度学场景 |
|------|------|-------------|--------------|
| `vecadd_128.mlir` | 向量加法 | `linalg.generic` + `arith.addf` | 残差连接 |
| `relu_4x4.mlir` | ReLU 激活 | `linalg.generic` + `arith.cmpf` + `select` | 全模型通用 |
| `tanh_4.mlir` | Tanh 激活 | `linalg.generic` + `math.tanh` | RNN / LSTM |
| `softmax_4.mlir` | 指数运算 (exp) | `linalg.generic` + `math.exp` | Attention softmax |
| `broadcast_4x4.mlir` | 标量广播 | `linalg.generic` + `affine_map<()>` | Bias 加法 |

### 归约操作（2 个）

| 文件 | 操作 | MLIR 核心模式 | 对应场景 |
|------|------|-------------|---------|
| `reduce_sum_4x4.mlir` | 求和归约 | `linalg.generic` + `reduction` iter | Layer Norm |
| `reduce_max_4x4.mlir` | 最大值归约 | `linalg.generic` + `arith.cmpf` + `reduction` | Softmax 数值稳定 |

### 矩阵运算（2 个）

| 文件 | 操作 | MLIR 核心模式 | 对应场景 |
|------|------|-------------|---------|
| `matmul_4x4x4.mlir` | 矩阵乘法 | `linalg.matmul` (named op) | Linear/MLP |
| `depthwise_conv_4x4.mlir` | 深度卷积 | `linalg.depthwise_conv_2d_nhwc_hwcm` | MobileNet |

### 融合操作（1 个）

| 文件 | 操作 | MLIR 核心模式 | 对应场景 |
|------|------|-------------|---------|
| `fused_128.mlir` | add + mul 融合 | 连续 `linalg.generic` × 2 | 算子融合演示 |

### 降级验证

| 用例 | 输入行数 | Lower 到 LLVM | 膨胀率 |
|------|---------|--------------|--------|
| vecadd_128 | 3 行 | 38 行 | 12.7× |
| matmul_4x4x4 | 1 行 | 74 行 | **74×** |
| relu_4x4 | 5 行 | 42 行 | 8.4× |
| softmax_4 | 5 行 | 17 行 | 3.4× |
| tanh_4 | 4 行 | 17 行 | 4.3× |
| reduce_sum_4x4 | 5 行 | 37 行 | 7.4× |
| reduce_max_4x4 | 5 行 | 47 行 | 9.4× |
| broadcast_4x4 | 5 行 | 24 行 | 4.8× |
| depthwise_conv_4x4 | 3 行 | 113 行 | 37.7× |
| fused_128 | 15 行 | 59 行 | 3.9× |

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

### 运行对比

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
