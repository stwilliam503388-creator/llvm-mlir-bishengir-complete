# ascend-npu-compiler-learning 完整学习计划

> Status: planning | Created: 2026-06-21 | Author: Hermes Agent
> 目标：将一个零基础学习者从"编译器是什么"带到"能看懂 Ascend NPU 编译器源码"

---

## 项目定位

一个面向 **零编译器基础** 的 AI 工程师的 Ascend NPU 编译器学习项目。

**学完四个阶段后**，学习者能够：
- 理解编译器的基本概念和工作原理
- 手写并运行 LLVM Pass
- 理解 MLIR 和 Dialect 系统
- 看懂 AscendNPU-IR 的源码结构和 Lowering 流程

**总预估时间**：约 8 小时（纯学习）+ 构建等待时间

---

## 整体架构

```
ascend-npu-compiler-learning/
├── README.md              ← 项目总入口
├── SUMMARY.md             ← 学习路径总览
├── setup.sh               ← 一键环境检查
├── .gitignore             ← 防编译产物误提交
│
├── docs/
│   ├── why-ascend.md      ← 为什么学（动机）
│   ├── quickstart.md      ← 2 小时快速入门
│   ├── glossary.md        ← 术语表（20+ 术语）
│   │
│   ├── primer/            ← Phase 1: 编译器入门
│   │   ├── README.md
│   │   ├── 00-编译器是什么.md
│   │   ├── 01-AST与IR.md
│   │   ├── 02-Pass与Lowering.md
│   │   └── 03-从Triton到Ascend.md
│   │
│   ├── llvm/              ← Phase 2: LLVM 动手
│   │   ├── README.md
│   │   ├── 00-环境搭建.md
│   │   ├── 01-LLVM-IR快速入门.md
│   │   ├── 02-第一个LLVM-Pass.md
│   │   └── 03-LLVM工具箱速览.md
│   │
│   ├── mlir/              ← Phase 3: MLIR 学习
│   │   ├── README.md                          ← 新写
│   │   ├── 00-从LLVM到MLIR.md                 ← 新写
│   │   ├── 01-Toy-Tutorial导读.md             ← 新写
│   │   ├── 02-ascendnpu-ir快速上手.md          ← 新写
│   │   └── MLIR-L00~L08 (9篇，包装旧文档)      ← 已有，加 frontmatter
│   │
│   └── ascend/            ← Phase 4: Ascend NPU 后端
│       ├── README.md                          ← 新写
│       ├── 00-Ascend-NPU硬件概述.md            ← 新写
│       ├── 01-husion-hivm-Dialect详解.md       ← 新写
│       ├── 02-一个Ascend-Pass详解.md           ← 新写
│       └── 03-构建与调试指南.md                ← 新写
│
├── projects/
│   ├── README.md
│   ├── hello-pass/        ← Phase 2 动手（已完成）
│   ├── mlir-hello/        ← Phase 3 动手（新建）
│   └── ascend-samples/    ← Phase 4 精选用例（新建，从 AscendNPU-IR 挑选 5 个）
│
├── references/
│   └── README.md          ← 外部资源索引
│
└── plans/                 ← 计划文档（本文件及历史计划）
```

---

## 四个阶段详细设计

### Phase 1: Primer — 编译器入门

**状态**：✅ 已完成（4 篇文档 + README）
**新增工作**：章节导航、时间预估、术语表链接、自检清单（已部分完成）

**文档清单**：

| # | 文档 | 核心概念 | 时间 |
|---|------|---------|------|
| 00 | 编译器是什么 | 编译器 vs 解释器、前后端 | 10 min |
| 01 | AST 与 IR | 抽象语法树、中间表示 | 10 min |
| 02 | Pass 与 Lowering | 编译优化、Pass 机制 | 15 min |
| 03 | 从 Triton 到 Ascend | AI 编译器实战 | 10 min |

**衔接设计**：
- 03 末尾 → Phase 2 LLVM
- 02 末尾 → hello-pass 动手

---

### Phase 2: LLVM — 动手写 Pass

**状态**：✅ 已完成（4 篇文档 + 1 个项目）
**新增工作**：hello-pass 项目（已构建验证）、环境搭建指南

**文档清单**：

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | 环境搭建 | macOS/Linux 装好 LLVM | 15 min |
| 01 | LLVM IR 快速入门 | 读懂 IR，自己生成 | 30 min |
| 02 | 第一个 LLVM Pass | 理解 Pass，完成 3 个挑战 | 40 min |
| 03 | LLVM 工具箱速览 | 知道有哪些工具 | 10 min |

**动手项目**：

| 项目 | 说明 |
|------|------|
| `projects/hello-pass/` | 30 行代码，一键运行，打印函数信息 |

**验证**：macOS LLVM 22.1.6 下 `./run.sh` 通过。

**衔接设计**：
- 03 末尾 → Phase 3 MLIR

---

### Phase 3: MLIR — 多层中间表示

**状态**：🚧 计划中
**新增工作**：3 篇桥接文档 + 包装 9 篇旧文档 + 1 个动手项目

**文档清单**：

| # | 文档 | 目标 | 时间 | 来源 |
|---|------|------|------|------|
| 00 | 从 LLVM 到 MLIR | LLVM→MLIR 概念对照、Dialect 解剖 | 15 min | 新写 |
| 01 | Toy Tutorial 导读 | 跑通 MLIR 官方教程 Ch1-3 | 60 min | 新写 |
| 02 | ascendnpu-ir 快速上手 | 项目结构、核心 Dialect 初探 | 30 min | 新写 |
| - | MLIR-L00 (速通与AscendNPU-IR实战) | 5 步 roadmap | 30 min | 包装 |
| - | MLIR-L01 (ToyTutorial Ch1-2) | AST → MLIR Dialect | 30 min | 包装 |
| - | MLIR-L02 (ToyTutorial Ch3-6) | Pass 优化 + 代码生成 | 40 min | 包装 |
| - | MLIR-L03 (自定义Pass实战) | 手写 op-counter Pass | 30 min | 包装 |
| - | MLIR-L04~L08 | Standalone/ToyMini/Triton 分析等 | 各 15-30 min | 包装 |

**旧文档包装方式**（每篇 +5 行 frontmatter）：
```markdown
> 📍 Phase 3 MLIR | [返回入口](./README.md)
> 前置：[00-从LLVM到MLIR](./00-从LLVM到MLIR.md)
> 预估时间：XX 分钟
```

**动手项目**：

| 项目 | 说明 |
|------|------|
| `projects/mlir-hello/` | MLIR 版 HelloPass，对标 hello-pass |

**mlir-hello 和 hello-pass 对照**：

| | hello-pass (LLVM) | mlir-hello (MLIR) |
|---|---|---|
| 遍历单位 | `Function &F` | `func::FuncOp` |
| 回调 | `run(Function &F)` | `runOnOperation()` |
| 打印 | `F.getName()` | `func.getName()` |
| 统计 | `BB.size()` | `func.walk()` |
| 注册 | `llvmGetPassPluginInfo()` | `mlirGetPassPluginInfo()` |
| 运行 | `opt --passes="hello"` | `mlir-opt --pass-pipeline="hello-mlir"` |

**衔接设计**：
- 02 末尾 → Phase 4 Ascend

---

### Phase 4: Ascend NPU — 编译器后端实战

**状态**：🚧 计划中
**新增工作**：4 篇文档 + 1 个精选用例集

**文档清单**：

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | Ascend NPU 硬件概述 | Da Vinci 架构、NPU vs GPU | 10 min |
| 01 | husion/hivm Dialect 详解 | 融合→指令 完整 Lowering | 20 min |
| 02 | 一个 Ascend Pass 详解 | 3 部分拆解 + 测试对照 | 25 min |
| 03 | 构建与调试指南 | 环境→构建→测试→常见坑 | 30 min |

**动手项目**：

| 项目 | 说明 |
|------|------|
| `projects/ascend-samples/` | 从 AscendNPU-IR 131 个测试中精选 5 个关键用例 |

**5 个精选用例的递进关系**：

```
用例 1: simple-add          ← 看懂一个 linalg.generic 长什么样
    ↓                          (14行 linalg → 3行 husion)
用例 2: fusion-add-mul       ← 两个操作融合为一个
    ↓                          (30行 linalg → 5行 husion)
用例 3: husion-to-hivm       ← 融合 IR → 虚拟指令
    ↓                          (husion.elemwise → hivm.vadd)
用例 4: hivm-to-llvm        ← 虚拟指令 → LLVM IR
    ↓                          (回到熟悉的 LLVM)
用例 5: full-pipeline        ← 一条 add 走完全程
    ↓                          (--mlir-print-ir-after-all)
```

每个用例结构：
```
NN-title/
├── input.mlir        ← 输入 IR
├── expected.mlir     ← 预期输出 IR
└── README.md         ← 逐行解读 + 运行命令
```

**用例与文档的对应**：

| 用例 | 对应文档 |
|------|---------|
| 01-simple-add | ascend/01 开头 |
| 02-fusion-add-mul | ascend/01 融合节 |
| 03-husion-to-hivm | ascend/01 hivm 节 |
| 04-hivm-to-llvm | ascend/01 末尾 |
| 05-full-pipeline | ascend/02 |

---

## 基础设施

| 文件 | 内容 | 状态 |
|------|------|:--:|
| `README.md` | 项目总入口、4 阶段总览、快速开始 | ✅ |
| `SUMMARY.md` | 完整学习目录、每阶段文档清单 | ✅ |
| `setup.sh` | 一键依赖检查（llvm-config, cmake, python3, git） | ✅ |
| `.gitignore` | 防编译产物误提交 | ✅ |
| `docs/why-ascend.md` | 为什么学（5 场景 + CUDA vs Ascend 对比表） | ✅ |
| `docs/quickstart.md` | 从零到写 Pass 的 2 小时路线 | ✅ |
| `docs/glossary.md` | 20+ 术语中英对照（核心/LLVM工具/文件格式/Ascend） | ✅ |
| `references/README.md` | 官方文档 + 推荐书籍 | ✅ |

---

## 学习连通性设计

### 全局导航链

```
README.md
  ├→ why-ascend.md (动机)
  ├→ quickstart.md (快速入门)
  │    └→ primer/00 → 01 → 02 → 03
  │                          ↓
  │                    hello-pass (动手)
  │                          ↓
  │                    llvm/00 → 01 → 02 → 03
  │                                      ↓
  │                              mlir-hello (动手)
  │                                      ↓
  │                              mlir/00 → 01 → 02 → MLIR-Lxx
  │                                                  ↓
  │                                        ascend/00 → 01 → 02 → 03
  │                                                  ↓
  │                                        ascend-samples (动手)
  │
  ├→ glossary.md (遇到术语随时回来查)
  └→ references/ (外部资源)
```

### 每篇文档底部统一元素

```markdown
> 📖 遇到不认识的术语？→ [术语表](../glossary.md)
> **下一步**：[下一章标题](./next-chapter.md)
```

---

## 执行阶段（分 3 批）

### 批次 1：Phase 1 + Phase 2 + 基础设施（✅ 已完成）

| 类别 | 文件数 | 状态 |
|------|--------|:--:|
| 文档 | 8 篇 + 4 个 README | ✅ |
| 项目 | hello-pass | ✅ |
| 基础设施 | why-ascend, quickstart, glossary, references, .gitignore, setup.sh | ✅ |
| 全局连接 | 章节导航、术语表链接、自检清单 | ✅ |

### 批次 2：Phase 3 文档 + mlir-hello 项目（🚧 待执行）

| 类别 | 文件数 | 来源 |
|------|--------|------|
| 桥接文档 | 3 篇 + README | 新写 |
| 旧文档包装 | 9 篇 | 加 frontmatter |
| 动手项目 | mlir-hello（5 文件） | 新建 |
| 全局更新 | llvm/03 链接、SUMMARY.md、quickstart.md | 修改 |

### 批次 3：Phase 4 文档 + ascend-samples 项目（🚧 待执行）

| 类别 | 文件数 | 来源 |
|------|--------|------|
| 文档 | 4 篇 + README | 新写 |
| 动手项目 | ascend-samples（16 文件） | 新建 |
| 全局更新 | quickstart.md、SUMMARY.md、README.md、projects/README.md | 修改 |

---

## 验证策略

### 每批次完成后的验证

| 批次 | 验证项 |
|------|--------|
| 1 | hello-pass `./run.sh` 通过（✅ 已验证 LLVM 22.1.6） |
| 2 | mlir-hello `./run.sh` 通过（macOS LLVM 22） |
| 3 | ascend-samples 用例 README 中的命令可执行 |

### 全项目链接验证

`find docs -name "*.md" | xargs grep "\.md)"` 无断链。

### 学习者路径验证

一个零基础读者，从 README 开始，跟着导航链走完 4 个阶段，中间不会迷路、不会遇到死胡同。

---

## 设计决策记录

| 决策 | 理由 |
|------|------|
| Phase 3 复用旧 MLIR-L 文档 | 尊重原作者劳动，不重复造轮子 |
| Phase 3 新增 3 篇桥接而非修改旧文档 | 桥接是"翻译层"，旧文档是"内容层"，各司其职 |
| Phase 4 不新建 fusion-demo 和 ascend-trace | AscendNPU-IR 已有 131 个测试用例，精选 5 个即可 |
| ascend-samples 依赖 AscendNPU-IR 构建 | 学习者迟早要 clone 真实项目，不如在这里就引入 |
| mlir-hello 对标 hello-pass 的结构 | 降低认知跳跃，LLVM→MLIR 对照表让迁移变简单 |
| 每篇文档 ≤300 行 | 初学者消化极限 |
| 每个项目独立可运行 | 降低环境门槛，先跑通再理解 |
