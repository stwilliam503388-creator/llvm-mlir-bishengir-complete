> 📍 Phase 3 MLIR | [返回入口](./README.md)
> 前置：[00-从LLVM到MLIR](./00-从LLVM到MLIR.md)
> 预估时间：10 min

---
created: 2026-06-21
tags: [ascendnpu-ir, demo, mlir, ascend]
aliases: [ascendnpu-ir demo, AscendNPU-IR 可运行 demo]
---

# ascendnpu-ir-demo：可运行的 MLIR 降级流水线

> 用标准 `mlir-opt` 模拟 AscendNPU-IR 三阶段降级（Linalg → HFusion → HIVM）。
> 所有用例在当前 Mac 上可直接运行。

---

## 项目位置

```text
~/hermes-workspace/ascendnpu-ir/ascendnpu-ir-demo/
├── README.md
├── ascendnpu-ir-demo.py              — Python 生成器
├── run-demo.sh                    — 批量运行脚本
├── test-cases/
│   ├── vecadd_128.mlir            — 向量加法
│   ├── matmul_4x4x4.mlir          — 矩阵乘法
│   └── fused_128.mlir             — 融合操作
└── results/                       — 运行结果
```

---

## 验证结果

所有用例通过 `mlir-opt` 验证：

| 用例 | 输入行数 | Stage1 (affine) | Stage3 (llvm) | 倍数 |
|------|---------|----------------|---------------|------|
| vecadd | 3 行 | 18 行 | 38 行 | 12.7x |
| matmul | 1 行 | 18 行 | **72 行** | **72x** |
| fused  | 15 行 | 20 行 | 59 行 | 3.9x |

matmul 膨胀 72x 的原因：三个嵌套循环被展开为 `scf.for` + `affine.load/store` + `arith.addf/mulf`。

---

## AscendNPU-IR 对应

```mlir
// AscendNPU-IR 输入 (test-cases/vecadd_128.mlir):
module {
  func.func @vecadd(%A, %B, %C) {
    linalg.generic { arith.addf }     ← 同 AscendNPU-IR linalg-to-hfusion.mlir
  }
}

// bishengir 输出 (模拟，实际需 bishengir-opt):
// Pass1:  linalg.generic → hfusion.elemwise_binary {fun = add}
// Pass2:  hfusion.elemwise_binary → hivm.load + hivm.vadd + hivm.store

// 本 demo 输出:
// Stage1: affine.for + affine.load/store + arith.addf
// Stage3: llvm.func + llvm.load/store + llvm.add
```

---

## 运行方式

```bash
# 跑单个用例
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
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


> 📖 [术语表](../glossary.md)