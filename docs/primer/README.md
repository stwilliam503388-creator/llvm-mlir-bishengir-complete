# 编译器零基础入门（Primer）

> 写给 AI 工程师的编译器概念速成，从 Triton 用户角度出发。

如果你还不确定为什么要学，先读：[为什么学 Ascend NPU 编译器？](../why-ascend.md)。

> 💡 遇到不认识的术语：查 [术语表](../glossary.md) 或 [技术术语速查手册](../reference/技术术语速查手册.md)。

## 阅读顺序

```text
00 — 编译器是什么            # 编译器三步工作法
  └─→ 01 — AST 与 IR          # 代码的两种中间形态
       └─→ 02 — Pass 与 Lowering
            └─→ 03 — 动手看 MLIR 长什么样
                 └─→ 04 — 从 Triton 到 Ascend
```

| 顺序 | 文档 | 读完后应该能 |
|---|---|---|
| 00 | [编译器是什么](./00-编译器是什么.md) | 说清前端、优化、中后端的大致分工 |
| 01 | [AST 与 IR](./01-AST与IR.md) | 区分源码、AST、IR、SSA |
| 02 | [Pass 与 Lowering](./02-Pass与Lowering.md) | 理解分析 Pass、转换 Pass、Lowering |
| 03 | [动手看 MLIR 长什么样](./03-动手看MLIR长什么样.md) | 看懂一个简单 MLIR 文件和 `mlir-opt` 输出 |
| 04 | [从 Triton 到 Ascend](./04-从Triton到Ascend.md) | 画出 Triton → MLIR → AscendNPU-IR 的大致路径 |

## 读完 Primer 之后

| 下一步 | 做什么 | 文件位置 |
|---|---|---|
| 1 | 跑第一个 LLVM Pass | [projects/hello-pass](../../projects/hello-pass/) |
| 2 | 看一个 MLIR 向量加法用例 | [01_vecadd.mlir](../../projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir) |
| 3 | 跑综合 demo 测试 | [projects/ascendnpu-ir-demo/run-tests.sh](../../projects/ascendnpu-ir-demo/run-tests.sh) |
| 4 | 继续学习 LLVM | [docs/llvm/README.md](../llvm/README.md) |
| 5 | 继续学习 MLIR | [docs/mlir/README.md](../mlir/README.md) |

## 概念对应的项目位置

| 概念 | 对应文件 |
|---|---|
| AST | `projects/toy-mini/toymini.cpp` |
| IR / SSA | `projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir` |
| 分析 Pass | `projects/ascendnpu-ir-op-counter/BishengirOpCounter.cpp` |
| 转换 Pass | `projects/ascendnpu-ir-op-counter/BishengirPeelTranspose.cpp` |
| Lowering | `projects/ascendnpu-ir-demo/run-demo.sh` |
| Dialect | `projects/standalone-mlir/include/standalone/StandaloneOps.td` |
