# ascendnpu-ir-demo — 可运行 MLIR 降级流水线

本项目用标准 `mlir-opt` 模拟 AscendNPU-IR 的核心思想：从高层 Linalg 语义逐步 lowering，并观察“过早展开为标量循环”和“保留硬件高级语义”之间的差异。

```text
AscendNPU-IR: Linalg → HFusion/Husion → HIVM → CANN/LLVM
标准 MLIR:    Linalg → Affine/SCF/CF → LLVM dialect
```

## 推荐从这里开始

先看最小向量加法：

```text
test-cases/mlir/01_basic/01_vecadd.mlir
```

它对应 AI 模型中的 residual add / shortcut，是理解 `linalg.generic`、SSA 值、`arith.addf` 和 lowering 的最小入口。

## 测试用例

| 类型 | 数量 | 路径 | 说明 |
|---|---:|---|---|
| MLIR 用例 | 31 | `test-cases/mlir/` | 按难度分为 basic / intermediate / advanced |
| Triton 对照 | 28 | `test-cases/triton/` | 与主要 MLIR 用例对应的 Python kernel；3 个 legacy MLIR 用例复用现有 Triton 对照 |
| matmul 优化变体 | 4 | `variants/` | baseline / tiling / vectorize / hardware mapping |

### MLIR 分级

| 难度 | 数量 | 目录 | 涉及概念 | 示例 |
|---|---:|---|---|---|
| ⭐ 入门 | 10 | `test-cases/mlir/01_basic/` | `linalg.generic`、逐元素运算、broadcast、fill | vecadd、relu、tanh、fused |
| ⭐⭐ 进阶 | 11 | `test-cases/mlir/02_intermediate/` | reduction、组合模式、条件分支 | sigmoid、silu、reduce_sum、layer_norm |
| ⭐⭐⭐ 复杂 | 10 | `test-cases/mlir/03_advanced/` | matmul、conv、pooling、多步 pipeline | matmul、conv2d、batch_norm |

## 快速运行

```bash
cd projects/ascendnpu-ir-demo

# 自动化测试：有 mlir-opt 时执行 RUN；无 mlir-opt 时检查标注
bash run-tests.sh

# 只看某类用例
bash run-tests.sh vecadd
bash run-tests.sh advanced

# 展示每个用例的三阶段 lowering，并把结果保存到 results/
bash run-demo.sh
```

如 `mlir-opt` 不在 PATH 中：

```bash
export MLIR_OPT=/path/to/mlir-opt
bash run-tests.sh
```

## 自动化测试设计

每个 `.mlir` 文件自带 LLVM lit / FileCheck 风格标注：

```mlir
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
```

`run-tests.sh` 支持两种模式：

- **完整模式**：发现 `mlir-opt` 后执行所有 `// RUN:` 命令。
- **轻量模式**：找不到 `mlir-opt` 时，只验证用例是否包含 `// RUN:` 标注，适合 CI 或文档检查环境。

## matmul 优化方案对比

```bash
bash variants/compare.sh
```

| Variant | 策略 | 核心思想 |
|---|---|---|
| V0 | baseline | 直接 lowering，容易展开为大量标量循环 |
| V1 | tiling | 增加 tile 循环层，改善局部性 |
| V2 | vectorize | 引入向量语义，减少标量指令 |
| V3 | hardware mapping | 保留 `matmul` 高级语义，类比映射到 Ascend Cube / `hivm.mmul` |

关键结论：NPU 编译器通常不希望把 matmul / conv 过早展开成标量循环，而是尽量保留高级语义，直到能映射到硬件矩阵乘或向量指令。

## 目录结构

```text
ascendnpu-ir-demo/
├── README.md
├── TESTING.md
├── LIMITATIONS.md
├── run-tests.sh
├── run-demo.sh
├── ascendnpu-ir-demo.py
├── test-cases/
│   ├── mlir/
│   │   ├── 01_basic/
│   │   ├── 02_intermediate/
│   │   ├── 03_advanced/
│   │   └── MAPPING.md
│   └── triton/
│       ├── 01_basic/
│       ├── 02_intermediate/
│       ├── 03_advanced/
│       └── MAPPING.md
└── variants/
    ├── compare.sh
    ├── variant0_baseline.sh
    ├── variant1_tiling.sh
    ├── variant2_vectorize.sh
    └── variant3_hw_mapping.sh
```

## 与其他文档的关系

- Primer 入门：[../../docs/primer/README.md](../../docs/primer/README.md)
- MLIR 学习：[../../docs/mlir/README.md](../../docs/mlir/README.md)
- Ascend 后端：[../../docs/ascend/README.md](../../docs/ascend/README.md)
- 用例导读：[../../docs/ascendnpu-ir-demo-case-guide.md](../../docs/ascendnpu-ir-demo-case-guide.md)
- Triton 对照映射：[test-cases/triton/MAPPING.md](test-cases/triton/MAPPING.md)
