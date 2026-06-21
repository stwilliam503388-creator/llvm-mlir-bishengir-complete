# bishengir-demo — 模拟 bishengir 三阶段降级的可运行 Demo

## 快速开始

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# VecAdd: linalg → affine（对应 bishengir 的 LinalgToHFusion）
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir

# VecAdd: 完整降级到 LLVM（对应 bishengir 完整流水线）
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  test-cases/vecadd_128.mlir
```

## 测试用例

| 文件 | 操作 | 对应 bishengir | 行数变化 |
|------|------|---------------|---------|
| `vecadd_128.mlir` | 向量加法 | `LinalgToHFusion` → `HFusionToHIVM` | 3 → 18 → 38 |
| `matmul_4x4x4.mlir` | 矩阵乘法 | `linalg.matmul` → Cube 指令 | 1 → 18 → 72 |
| `fused_128.mlir` | add + mul 融合 | 融合优化演示 | 15 → 20 → 59 |

## bishengir ↔ 标准 MLIR 对照

```
bishengir:                       标准 MLIR (本 demo):
─────────────                    ─────────────────────
linalg.generic                   linalg.generic
    ↓ -convert-linalg-to-hfusion     ↓ --convert-linalg-to-affine-loops
hfusion.elemwise_binary          affine.for + arith.addf
    ↓ -convert-hfusion-to-hivm      ↓ --lower-affine --convert-scf-to-cf
hivm.load/vadd/store             scf.for + cf.br
    ↓ CANN                           ↓ --convert-func-to-llvm
Ascend NPU 可执行代码              LLVM IR (可通过 lli 执行)
```

## 文件结构

```
bishengir-demo/
├── README.md                      ← 本文件
├── bishengir-demo.py              ← Python 生成器（可生成任意规模用例）
├── run-demo.sh                    ← 批量运行脚本
├── test-cases/                    ← MLIR 测试用例
│   ├── vecadd_128.mlir
│   ├── matmul_4x4x4.mlir
│   └── fused_128.mlir
└── results/                       ← 运行结果（自动生成）
