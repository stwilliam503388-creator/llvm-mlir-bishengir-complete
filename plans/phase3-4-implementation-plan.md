# Phase 3 & Phase 4 实施计划

> Status: executed | Created: 2026-06-21

---

## 现状

| 资源 | 内容 | 规模 |
|------|------|------|
| [MLIR 官方 Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) | MLIR 标准入门教程，从定义语言到生成代码 | 7 章，含完整代码 |
| [ascendnpu-ir](https://github.com/stwilliam503388-creator/ascendnpu-ir) | 用户的 MLIR 学习项目（AscendNPU-IR 的 fork） | 750 个源文件，含 docs/ |
| [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) | 华为官方 Ascend MLIR 编译器项目 | 750 个源文件，含 docs/ |

两个 ascendnpu-ir 仓库结构相同，是同一套代码的不同 fork。它们本身就是一个完整的 MLIR-based 编译器后端项目。

---

## 策略：不全量翻译，写"连接桥"

MLIR Toy Tutorial 和 AscendNPU-IR 的官方文档已经很好了（中英文都有）。本项目不需要重复翻译它们，而是写**桥接文档**——把 Phase 2 学到的 LLVM 概念映射到 MLIR/Ascend 世界。

```
Phase 2 (LLVM)           Phase 3 (MLIR)              Phase 4 (Ascend)
    ↓                        ↓                            ↓
 LLVM Pass            MLIR Toy Tutorial           AscendNPU-IR 源码
 LLVM IR          ←── 概念桥接文档 ──→        ←── 源码导读文档 ──→
 HelloPass              ascendnpu-ir 项目           构建 + 调试指南
```

---

## Phase 3: MLIR 入门（≈5 个文件）

### 任务 3.1: MLIR 概念桥接

**文件**: `docs/mlir/00-从LLVM到MLIR.md`

**目标**: 用 LLVM 概念解释 MLIR，降低认知跳跃

**内容大纲**:
```
1. MLIR 解决什么问题？
   - LLVM 只有一层 IR → 所有优化都在同一层做
   - MLIR 有多层 IR → 每层做最适合那层的优化
   - 类比：LLVM = 所有人都说英语；MLIR = 数学家用数学符号，司机用路标

2. 关键概念对照表
   | LLVM 概念 | MLIR 对应 | 区别 |
   | Pass | Pass | 基本相同 |
   | Function/BasicBlock | Operation/Region/Block | MLIR 更通用 |
   | .ll 文件 | .mlir 文件 | 语法不同 |
   | llvm::FunctionPass | mlir::Pass | 接口不同 |

3. MLIR 的核心创新：Dialect
   - Dialect = 一组自定义的 Operation 和 Type
   - 例如：linalg dialect 描述矩阵运算，scf dialect 描述循环
   - 对比：LLVM 只有一套固定的指令集

4. 一个 MLIR 文件的解剖
   - func.func, arith.addi, scf.for — 不同 dialect 的操作混用
   - 和 .ll 文件的对比
```

**验证**: 读者能用 LLVM 概念描述 MLIR 的对应物

### 任务 3.2: MLIR Toy Tutorial 导读

**文件**: `docs/mlir/01-Toy-Tutorial导读.md`

**目标**: 引导读者跑通 MLIR 官方 Toy Tutorial 的第 1-3 章

**内容大纲**:
```
1. Toy 语言是什么？
   - 一个极简的张量计算语言
   - def multiply_transpose(a, b) { return transpose(a) * transpose(b); }

2. 环境准备（macOS）
   - brew install llvm（已有，MLIR 在 LLVM 包里）
   - 确认 mlir-opt 可用

3. Chapter 1-3 导读
   - Ch1: 定义 Toy 语言的 AST 和 MLIR Dialect
   - Ch2: 用 MLIR 表示 Toy 程序
   - Ch3: 写 Pass 优化 Toy IR（消除冗余 transpose）

4. 每章学完后应能回答的问题
   - Ch1: 为什么要定义自己的 Dialect？
   - Ch2: Operation 和 Function 的区别？
   - Ch3: 这个 Pass 做了什么优化？为什么有效？
```

**验证**: 读者能独立编译并运行 Toy Tutorial Ch1-3

### 任务 3.3: ascendnpu-ir 项目初探

**文件**: `docs/mlir/02-ascendnpu-ir快速上手.md`

**目标**: 在 Toy Tutorial 之后，引导读者进入真实的 MLIR 项目

**内容大纲**:
```
1. ascendnpu-ir 是什么？
   - 一个基于 MLIR 的 Ascend NPU 编译器后端
   - 包含自定义 Dialect：husion（昇腾融合 IR）、hivm（昇腾指令集）

2. 项目结构速览
   - bishengir/ — 核心编译器代码
   - docs/ — 官方文档（中英文）
   - third-party/ — LLVM/MLIR 依赖

3. 构建与运行
   - cmake + ninja 构建
   - 运行测试

4. 关键文件导读
   - 找一个最小的 Dialect 定义（.td 文件）
   - 找一个最小的 Pass（.cpp 文件）
   - 对比 Toy Tutorial 和 ascendnpu-ir 的异同
```

**验证**: 读者能构建 ascendnpu-ir 并理解项目结构

---

## Phase 4: Ascend NPU 后端实战（≈4 个文件）

### 任务 4.1: Ascend NPU 硬件概念

**文件**: `docs/ascend/00-Ascend-NPU硬件概述.md`

**目标**: 理解 NPU 的基本计算模型，不需要硬件细节

**内容大纲**:
```
1. NPU vs GPU vs CPU
   - CPU: 少数复杂核心（控制流强）
   - GPU: 大量简单核心（并行计算）
   - NPU: 专用矩阵乘法单元（AI 推理最优）

2. Ascend NPU 的计算模型
   - Da Vinci 架构核心概念（Cube Unit / Vector Unit）
   - 数据搬运：L1/L2/HBM 三级缓存
   - 类比：厨房的切菜台(Cube)、炒锅(Vector)、冰箱(HBM)

3. 编译器需要做什么？
   - 把 AI 框架的算子（Conv、MatMul）翻译成 NPU 能执行的指令
   - 管理数据搬运（什么时候从 HBM 搬到 L1）
   - 这就是 ascendnpu-ir 在做的事
```

**验证**: 读者能用一句话解释"NPU 编译器后端在做什么"

### 任务 4.2: AscendNPU-IR 核心 Dialect 详解

**文件**: `docs/ascend/01-husion-hivm-Dialect详解.md`

**目标**: 深入理解 ascendnpu-ir 的两个核心 Dialect

**内容大纲**:
```
1. husion Dialect — 昇腾融合 IR
   - 为什么需要融合？减少数据搬运 = 提升性能
   - 核心操作：husion.elemwise_binary、husion.matmul 等
   - 从 linalg.generic 到 husion 的 Lowering

2. hivm Dialect — 昇腾指令集
   - NPU 可以执行的具体指令
   - 核心操作：hivm.vadd、hivm.load、hivm.store 等
   - 从 husion 到 hivm 的 Lowering

3. 完整的 Lowering 流程
   linalg.generic → husion.elemwise_binary → hivm.vadd
   每一步都做了什么？为什么需要这么多步？
```

**验证**: 读者能画出从 linalg 到 hivm 的 Lowering 路径

### 任务 4.3: 源码导读 — 一个 Pass 的完整生命周期

**文件**: `docs/ascend/02-一个Ascend-Pass详解.md`

**目标**: 选一个真实的 ascendnpu-ir Pass，逐行解读

**内容大纲**:
```
1. 选一个最简单的 Pass（如 ConvertLinalgToHusion）
2. 逐行解读
   - .td 文件：Dialect 和 Operation 的定义
   - .cpp 文件：Pass 的实现逻辑
   - 对应的测试 .mlir 文件：输入和预期输出
3. 读者动手
   - 加一行 debug 打印
   - 修改一个 Lowering 规则
   - 运行测试看效果
```

**验证**: 读者能修改一个 Pass 并看到效果变化

### 任务 4.4: 构建与调试指南

**文件**: `docs/ascend/03-构建与调试指南.md`

类似 `docs/llvm/00-环境搭建.md`，但针对 ascendnpu-ir 项目。

---

## 文件产出汇总

| # | 文件 | Phase | 类型 | 预估行数 |
|---|------|-------|------|---------|
| 1 | `docs/mlir/00-从LLVM到MLIR.md` | 3 | 新建 | ~200 |
| 2 | `docs/mlir/01-Toy-Tutorial导读.md` | 3 | 新建 | ~200 |
| 3 | `docs/mlir/02-ascendnpu-ir快速上手.md` | 3 | 新建 | ~200 |
| 4 | `docs/ascend/00-Ascend-NPU硬件概述.md` | 4 | 新建 | ~150 |
| 5 | `docs/ascend/01-husion-hivm-Dialect详解.md` | 4 | 新建 | ~250 |
| 6 | `docs/ascend/02-一个Ascend-Pass详解.md` | 4 | 新建 | ~250 |
| 7 | `docs/ascend/03-构建与调试指南.md` | 4 | 新建 | ~100 |

**总计**: 7 个文件，~1350 行。不需要翻译现有文档，只写"桥接"。

---

## 与 Phase 1/2 的衔接

```
Phase 1 (Primer) → Phase 2 (LLVM) → Phase 3 (MLIR) → Phase 4 (Ascend)
    现有 ✅             现有 ✅          本计划             本计划

桥接点：
- Phase 2 最后一篇（03-LLVM工具箱）末尾加"→ Phase 3: MLIR 入门"
- Phase 3 第一篇（00-从LLVM到MLIR）开头加"← 来自 Phase 2"
```

---

## 设计原则

| 原则 | 应用 |
|------|------|
| 不翻译已有文档 | MLIR Toy Tutorial 和 AscendNPU-IR 官方文档已足够好 |
| 写桥接，不写替代 | 本文档帮你"到达"官方文档，不是替代它 |
| 保持 Phase 1/2 风格 | 比喻、对照表、验证清单、时间预估 |
| 优先可运行 | 每篇配有可执行的命令，不是纯阅读 |
