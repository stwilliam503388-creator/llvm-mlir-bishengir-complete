# 编译器零基础入门（Primer）

> 写给 AI 工程师的编译器概念速成，从 Triton 用户角度出发。

> 💡 **遇到不认识的术语？** 查 `docs/reference/技术术语速查手册.md`，
> 298 条术语，每条含"一句话"+"类比"+"为什么重要"+"在项目中的位置"。

## 阅读顺序

```
00 — 为什么 AI 工程师要学编译器     ← 从 Triton 翻车场景开始 (8min)
  └─→ 01 — AST 和 IR              ← 代码的两种中间形态 (6min)
       └─→ 02 — Pass 和 Lowering   ← IR 是怎么变成机器码的 (6min)
            └─→ 03 — 动手看 MLIR   ← 亲手运行 mlir-opt 看降级过程 (10min)
                 └─→ 04 — 完整路径 ← 从 Triton 到 Ascend NPU (5min)
```

**总共约 35 分钟。**

## 读完 Primer 之后

| 步骤 | 做什么 | 文件位置 |
|------|--------|---------|
| 1 | 深入理解 SSA | `docs/llvm/L00-SSA.md` |
| 2 | 跑降级对比脚本 | `projects/ascendnpu-ir-demo/variants/compare.sh` |
| 3 | 了解 MLIR 设计理念 | `docs/mlir/L00-MLIR概述.md` |
| 4 | 看 MLIR ↔ Triton 双向映射 | `test-cases/triton/MAPPING.md` |

## 每个概念对应的项目位置

| 概念 | 对应文件 |
|------|---------|
| AST | `projects/toy-mini/toymini.cpp` → `struct NumberExpr` |
| IR | `projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir` |
| SSA | 同上 → `%0, %1, %2` |
| 分析 Pass | `projects/ascendnpu-ir-op-counter/BishengirOpCounter.cpp` |
| 转换 Pass | `projects/ascendnpu-ir-op-counter/BishengirPeelTranspose.cpp` |
| Lowering | `projects/ascendnpu-ir-demo/variants/compare.sh` |
| Dialect | `projects/standalone-mlir/StandaloneOps.td` |
