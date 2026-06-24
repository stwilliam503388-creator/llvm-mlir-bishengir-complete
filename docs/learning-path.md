# 完整学习路径与项目背景

本文承接根目录 `README.md` 中被压缩掉的长篇说明，把项目背景、读者定位、分阶段路线和工程项目之间的关系集中放在 `docs/` 下。根 `README.md` 保持短入口；需要教学细节时读本文。

## 为什么需要这个项目

AI 芯片正在从通用计算走向专用计算。以华为昇腾 Ascend 为代表的 NPU 在推理和训练场景中越来越重要，但它的软件栈比“写一个 kernel 然后运行”复杂得多。

一段 Triton kernel 进入 Ascend NPU 的执行路径，可以粗略理解为：

```text
Triton Python kernel
        │
        ▼
Triton IR / TT Dialect
        │
        ▼
MLIR Linalg / Arith / Func 等标准 dialect
        │
        ▼
AscendNPU-IR / BishengIR
  Linalg → HFusion/Husion → HIVM
        │
        ▼
CANN Runtime
        │
        ▼
Ascend NPU 执行
```

每一步都会遇到编译器概念：IR、SSA、Dialect、Operation、Pass、Pattern Rewriting、Lowering。现有资料常见两个断层：

- 只讲 LLVM/MLIR 基础，难以对应到真实 AI 编译器后端。
- 只讲 Triton 或 Ascend 应用，遇到底层 IR 和 pass 时难以继续阅读源码。

本仓库的目标是补上中间路径：从零基础编译器概念开始，逐步过渡到 LLVM、MLIR、AscendNPU-IR 和 Triton 对照。

## AscendNPU-IR 与 BishengIR

本仓库中出现的 **AscendNPU-IR** 与 **BishengIR** 指向同一类 Ascend NPU MLIR 编译器项目语境：

| 名称 | 说明 |
|---|---|
| AscendNPU-IR | 官方仓库和文档中使用的项目名称 |
| BishengIR | 源码、命名空间和工具名中常见的名称，如 `bishengir-opt` |
| HFusion / Husion | 表达融合后高层算子语义的 Ascend 自定义 dialect 语境 |
| HIVM | 更接近 NPU 指令/硬件抽象的 dialect 语境 |

典型 lowering 链路可以理解为：

```text
输入: Linalg / Arith / Func 等标准 MLIR dialect
        │
        ▼
Pass1: -convert-linalg-to-hfusion
        Linalg → HFusion/Husion
        │
        ▼
Pass2: -convert-arith-to-hfusion
        Arith → HFusion/Husion
        │
        ▼
Pass3: -convert-hfusion-to-hivm
        HFusion/Husion → HIVM
        │
        ▼
HIVM → CANN Runtime → Ascend NPU
```

`projects/ascendnpu-ir-demo/` 使用标准 `mlir-opt` 模拟这条路线，帮助你在没有 Ascend 硬件和完整 BishengIR 工具链时先建立 lowering 直觉。

## 适合谁读

| 读者 | 可以从哪里开始 |
|---|---|
| 没有编译器基础的 AI 工程师 | [docs/primer/](./primer/) |
| 会写 Triton，但看不懂 MLIR 的学习者 | [docs/quickstart.md](./quickstart.md) → [projects/ascendnpu-ir-demo/](../projects/ascendnpu-ir-demo/) |
| 想理解 LLVM Pass 的开发者 | [docs/llvm/](./llvm/) → [projects/hello-pass/](../projects/hello-pass/) |
| 想学习 MLIR dialect / lowering 的开发者 | [docs/mlir/](./mlir/) → [projects/standalone-mlir/](../projects/standalone-mlir/) |
| 想阅读 AscendNPU-IR / triton-ascend 源码的人 | [docs/ascendnpu-ir/](./ascendnpu-ir/)、[references/](../references/) |

前置知识不要求编译器经验；能读基础 C++、Python 和命令行输出即可。

## 分阶段学习路线

### Stage -1：编译器零基础入门

目标：建立 AST → IR → Pass → Lowering 的基本直觉。

| 步骤 | 文档 | 重点 | 对应项目 |
|---|---|---|---|
| -1.1 | `docs/primer/00-编译器是什么.md` | 前端、优化、中后端 | — |
| -1.2 | `docs/primer/01-AST与IR.md` | AST、IR、SSA | `projects/toy-mini/` |
| -1.3 | `docs/primer/02-Pass与Lowering.md` | 分析 Pass、转换 Pass、Lowering | `projects/ascendnpu-ir-op-counter/` |
| -1.4 | `docs/primer/04-从Triton到Ascend.md` | Triton 到 Ascend 的路径 | `projects/ascendnpu-ir-demo/` |

关键突破：知道“编译器不是一次翻译到底”，而是一层层维护语义并逐步 lowering。

### Stage 0：LLVM IR 基础

目标：理解 LLVM 的 SSA、类型系统、控制流和 Pass 模型，为 MLIR 打基础。

| 步骤 | 文档 | 知识点 | 产出 |
|---|---|---|---|
| 0.1 | `docs/llvm/LLVM-L00-速通总览.md` | LLVM 三段式架构 | 整体认知 |
| 0.2 | `docs/llvm/LLVM-L01-架构与HelloWorld.md` | Module / Function / BasicBlock | 能读 `.ll` 文件 |
| 0.3 | `docs/llvm/LLVM-L02-类型系统与GEP.md` | 类型、指针、GEP | 理解地址计算 |
| 0.4 | `docs/llvm/LLVM-L03-控制流与Phi节点.md` | CFG、`br`、φ 节点 | 理解控制流汇合 |
| 0.5 | `docs/llvm/LLVM-L05-Pass开发.md` | New Pass Manager | 能写简单 Pass |
| 0.6 | `docs/llvm/LLVM-L06-IR速查表.md` | 指令与调试命令 | 快速查阅 |

关键突破：理解 SSA 与 φ 节点。它们是理解 MLIR Region、Block 和 Value 的前置概念。

### Stage 1：MLIR 核心概念

目标：掌握 MLIR 的多层 IR 哲学，理解 dialect / operation / pass 三大概念。

| 步骤 | 文档 | 知识点 | 对应项目 |
|---|---|---|---|
| 1.1 | `docs/mlir/MLIR-L00-速通与AscendNPU-IR实战.md` | Operation、Dialect、Region | `projects/ascendnpu-ir-demo/` |
| 1.2 | `docs/mlir/MLIR-L01-ToyTutorial速通-Ch1-Ch2.md` | Toy Tutorial、TableGen | `projects/toy-mini/` |
| 1.3 | `docs/mlir/MLIR-L02-ToyTutorial速通-Ch3-Ch6.md` | Pattern Rewriting、ConversionTarget | `projects/ascendnpu-ir-op-counter/` |
| 1.4 | `docs/mlir/MLIR-L03-自定义AscendNPU-IR-Pass实战.md` | `walk` 与 rewrite pattern | `projects/ascendnpu-ir-op-counter/` |
| 1.5 | `docs/mlir/MLIR-L04-Standalone实战总结.md` | CMake、TableGen、独立 dialect | `projects/standalone-mlir/` |

关键突破：理解“多个 dialect 共存”不是复杂化，而是为了在不同阶段保留不同层级的语义。

### Stage 2：工程实战

目标：从读文档走向能运行、能修改、能验证。

| 步骤 | 项目 | 行动 | 验证 |
|---|---|---|---|
| 2.1 | `projects/ascendnpu-ir-demo/` | 跑 MLIR 用例，观察 lowering | `bash run-tests.sh` |
| 2.2 | `projects/ascendnpu-ir-demo/variants/` | 对比 matmul 的 4 种优化策略 | `bash variants/compare.sh` |
| 2.3 | `projects/toy-mini/` | 编译运行 Toy 解析器 | `g++ -std=c++17 ...` |
| 2.4 | `projects/standalone-mlir/` | 构建自定义 MLIR dialect | `standalone-opt` |
| 2.5 | `projects/ascendnpu-ir-op-counter/` | 阅读分析 Pass 与转换 Pass | 对照 Toy Tutorial |

关键突破：把“知道概念”变成“能用工具验证自己的理解”。

### Stage 3：体系对照

目标：把本仓库中的简化 demo 对标到真实项目。

| 步骤 | 入口 | 对标内容 | 产出 |
|---|---|---|---|
| 3.1 | `docs/mlir/MLIR-L06-TritonMLIR体系分析.md` | Triton TT / TritonGPU Dialect | 了解 Triton IR 层级 |
| 3.2 | `docs/mlir/MLIR-L07-triton-ascend后端深度分析.md` | Triton 到 Ascend 后端 | 看懂 Python 与 C++ 接口层 |
| 3.3 | `docs/ascend/01-husion-hivm-Dialect详解.md` | HFusion/Husion、HIVM | 对照 Ascend 自定义 dialect |
| 3.4 | `docs/ascendnpu-ir/translations/` | 官方文档翻译与分析 | 深入 pass / dialect 细节 |
| 3.5 | `references/ascendnpu-ir-mapping.md` | 本仓库与外部源码映射 | 定位真实源码文件 |

关键突破：理解 Triton + MLIR + AscendNPU-IR 如何组成完整的 Ascend 编译链路。

## 工程项目如何配合文档

| 项目 | 角色 | 推荐阅读时机 |
|---|---|---|
| `projects/hello-pass/` | 第一个 LLVM Pass，建立“Pass 会遍历 IR”的直觉 | LLVM 阶段初期 |
| `projects/opt-pass/` | 修改 IR 的 LLVM Pass，理解转换型 pass | LLVM 阶段后半 |
| `projects/mlir-hello/` | 第一个 MLIR Pass | MLIR 入门后 |
| `projects/toy-mini/` | 纯 C++17 Toy 前端，理解 AST 和 IR 生成 | Primer / MLIR Toy 阶段 |
| `projects/standalone-mlir/` | 自定义 MLIR dialect 工程模板 | MLIR dialect 阶段 |
| `projects/ascendnpu-ir-op-counter/` | AscendNPU-IR Pass 参考代码 | Pattern / Pass 阶段 |
| `projects/ascend-samples/` | Ascend lowering 精选用例 | Ascend 后端阶段 |
| `projects/ascendnpu-ir-demo/` | 综合 MLIR lowering demo | 全程反复使用 |

## 推荐最短闭环

如果只想快速建立完整路线感，可以按这个顺序：

```text
1. 读 docs/why-ascend.md
2. 读 docs/primer/README.md 和 00~04
3. 跑 projects/hello-pass/run.sh
4. 读 projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir
5. 跑 projects/ascendnpu-ir-demo/run-tests.sh
6. 读 docs/ascend/01-husion-hivm-Dialect详解.md
7. 回到 docs/mlir/ 和 docs/ascendnpu-ir/ 深入查缺补漏
```

这条路径的重点不是一次学完所有细节，而是先建立“源码 → IR → Pass → Lowering → 硬件语义”的完整地图。
