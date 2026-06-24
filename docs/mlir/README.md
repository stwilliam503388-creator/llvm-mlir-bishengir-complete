# Phase 3 — MLIR 学习

> 衔接 Phase 2（LLVM），用已学会的 LLVM 概念理解 MLIR
> 前置：[Phase 2 LLVM](../llvm/README.md)

## 学习路径

```
① LLVM→MLIR 概念对照 ──→ ② Toy Tutorial 带读 ──→ ③ ascendnpu-ir 上手
      (15 min)                (60 min)                  (30 min)
                                    ↓
                         ④ 旧文档深入 (MLIR-L00~L08，选读)
```

## 文档列表

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | [从 LLVM 到 MLIR](./00-从LLVM到MLIR.md) | LLVM→MLIR 概念对照，理解 Dialect | 15 min |
| 01 | [Toy Tutorial 导读](./01-Toy-Tutorial导读.md) | 跑通 MLIR 官方教程 Ch1-3 | 60 min |
| 02 | [ascendnpu-ir 快速上手](./02-ascendnpu-ir快速上手.md) | 进入真实的 MLIR 编译器项目 | 30 min |
| - | [MLIR-L00~L08](./MLIR-L00-速通与AscendNPU-IR实战.md) | 旧文档（实战深入，选读） | 各 15-30 min |

## 动手项目

| 项目 | 说明 |
|------|------|
| **[mlir-hello](../../projects/mlir-hello/)** | MLIR 版 HelloPass，45 行，一键运行 |

## 与 Phase 2 的衔接

| Phase 2 学会的 | Phase 3 怎么用到 |
|---------------|----------------|
| LLVM Pass（`run(Function &F)`） | MLIR Pass（`runOnOperation()`），思路一样 |
| LLVM IR（`.ll` 文件） | MLIR（`.mlir`），多了一个 Dialect 前缀 |
| hello-pass 项目 | mlir-hello 项目，结构对应 |

## 依托资源

- [MLIR 官方 Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)（7 章）
- [ascendnpu-ir](https://github.com/stwilliam503388-creator/ascendnpu-ir)（真实项目）
- [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR)（华为官方版本）

## 学完验证

- [ ] 能用 LLVM 概念解释 MLIR 的对应物
- [ ] 能跑通 Toy Tutorial Ch1-3
- [ ] 能说出 ascendnpu-ir 的核心 Dialect 和 Lowering 流程
