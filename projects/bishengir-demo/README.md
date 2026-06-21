# bishengir-demo — 可运行 MLIR 降级流水线

用标准 `mlir-opt` 模拟 AscendNPU-IR 三阶段降级（Linalg → HFusion → HIVM）。

---

## 测试用例

| 文件 | 操作 | 对应 AscendNPU-IR | 行数变化 |
|------|------|---------------|---------|
| `test-cases/vecadd_128.mlir` | 向量加法 | `LinalgToHFusion` → `HFusionToHIVM` | 3 → 18 → 38 行 |
| `test-cases/matmul_4x4x4.mlir` | 矩阵乘法 | `linalg.matmul` → Cube 指令 | 1 → 18 → **74 行** |
| `test-cases/fused_128.mlir` | add + mul 融合 | 融合优化演示 | 15 → 20 → 59 行 |

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

### V3 的核心思路 — AscendNPU-IR 实际采用的方案

```text
标准 MLIR 路径 (V0):
  linalg.matmul → affine.for×3 → scf.for+arith → llvm.load/add/mul/store  (74行)

AscendNPU-IR 路径 (≈V3):
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

## AscendNPU-IR ↔ 标准 MLIR 对照

```text
AscendNPU-IR:                      标准 MLIR (本 demo):
────────────────────             ────────────────────
linalg.generic                  linalg.generic
    ↓ -convert-linalg-to-hfusion    ↓ --convert-linalg-to-affine-loops
hfusion.elemwise_binary         affine.for + arith.addf
    ↓ -convert-hfusion-to-hivm     ↓ --lower-affine --scf-to-cf --func-to-llvm
hivm.load/vadd/store            llvm.load + llvm.add + llvm.store
```
