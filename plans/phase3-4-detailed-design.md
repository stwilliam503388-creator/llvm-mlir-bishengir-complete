# Phase 3 & 4 详细设计

> Status: planning | Created: 2026-06-21 | Updated: 2026-06-21

---

## 0. 发现：仓库已有旧 MLIR 文档

在 `docs/mlir/` 中存在 9 篇从未被 git 提交的旧文档（~1900 行）：

| 文件 | 行数 | 内容 |
|------|------|------|
| MLIR-L00-速通与AscendNPU-IR实战.md | 556 | MLIR 概念 + ascendnpu-ir 架构 + 5 步 roadmap |
| MLIR-L01-ToyTutorial速通-Ch1-Ch2.md | 288 | Toy 语言 AST → MLIR Dialect 定义 |
| MLIR-L02-ToyTutorial速通-Ch3-Ch6.md | 294 | Pass 优化 + 代码生成 |
| MLIR-L03-自定义AscendNPU-IR-Pass实战.md | 167 | 手写 op-counter Pass |
| MLIR-L04-Standalone实战总结.md | 81 | Standalone MLIR 项目模板 |
| MLIR-L05-ToyMini从零实现.md | 77 | 简化版 Toy 实现 |
| MLIR-L06-TritonMLIR体系分析.md | 182 | Triton 编译流程分析 |
| MLIR-L07-triton-ascend后端深度分析.md | 222 | Triton→Ascend 代码生成 |
| MLIR-L08-ascendnpu-ir-demo可运行流水线.md | 57 | 完整运行命令 |

**评估**：内容扎实，偏实战，但有三点不足：
1. 开头直接讲 MLIR 概念，没有从 LLVM 桥接——对刚学完 Phase 2 的读者跳跃太大
2. 文件名是英文缩写（"MLIR-L00"），不直观
3. 没有时间预估、验证清单、术语引导——缺乏 Phase 1/2 的打磨标准

---

## 1. 总体策略

```
旧 MLIR-L00~L08 (1900行，实战内容)
        +
Phase 3 桥接层（3篇新写，做"翻译官"）
        +
Phase 4 新写（4篇，填补 Ascend NPU 后端空白）
        =
完整 Phase 3 + Phase 4
```

**核心原则**：
- 旧文档不动内容，只做"包装"（加 frontmatter、导航、验证清单）
- Phase 3 新写 3 篇桥接：概念对照 → Toy 导读 → 项目上手
- Phase 4 全是新写：硬件 → Dialect → Pass → 构建
- 保持和 Phase 1/2 一致的风格：比喻、对照表、时间预估、自检清单

---

## 2. Phase 3 详细设计

### 架构

```
docs/mlir/
├── README.md              ← 新写：阶段入口（学习路线 + 文档索引）
├── 00-从LLVM到MLIR.md     ← 新写：概念桥接（LLVM→MLIR 对照表）
├── 01-Toy-Tutorial导读.md ← 新写：带读者跑通官方教程 Ch1-3
├── 02-ascendnpu-ir快速上手.md ← 新写：项目结构 + 核心 Dialect 初探
├── MLIR-L00~L08           ← 旧文档：实战内容（加 frontmatter 包装）
└── README.md              ← 已存在：旧 README
```

### 2.1 README.md（入口页，新写）

**目标**：读者打开后知道"Phase 3 是干什么的、按什么顺序学"

**内容**：
- 学习路线图（3 篇桥接 → 9 篇实战）
- 每篇文档的时间预估
- 与 Phase 2 的衔接表（LLVM Pass → MLIR Pass）
- 学完验证清单

### 2.2 00-从LLVM到MLIR.md（概念桥接，新写）

**目标**：用 Phase 2 学会的 LLVM 概念解释 MLIR

**大纲**（详细）：

```
1. MLIR 解决什么问题？（5 行）
   - LLVM 的痛点：只有一层 IR，所有优化挤在一起
   - MLIR 的方案：多层 IR，每层做最擅长的事
   - 类比：建筑工地的木工/电工/水管工 → 专用语言系统

2. LLVM → MLIR 概念对照表（核心）
   | LLVM | MLIR | 异同 |
   | Function | func.func Operation | MLIR 中函数也是 Operation |
   | BasicBlock | Block | 基本相同 |
   | Instruction | Operation | MLIR 更通用，可多返回值 |
   | Pass (FunctionPass) | Pass (mlir::Pass) | 接口不同，理念相同 |
   | .ll 文件 | .mlir 文件 | 语法不同，都有 SSA |
   | i32 / float | i32 / f32 | 兼容 |
   | %variable | %variable | SSA 变量命名相同 |

3. Dialect：MLIR 的核心创新
   - 类比：LLVM 是"所有人都说英语"；MLIR 是"数学家用数学符号，电工说电路图"
   - 常见 Dialect 速查：arith/scf/linalg/func/memref/llvm
   - 一个 .mlir 文件的解剖：func.func + arith.addi + scf.if 混用

4. 从 HelloPass 到 MLIR Pass
   - HelloPass: run(Function &F) → 打印信息
   - MLIR Pass: matchAndRewrite(Operation) → 模式匹配+替换
   - 对比表：相同思路，不同 API

5. 验证清单（5 题）
```

**关键设计决策**：
- 不解释 MLIR 怎么装（Phase 2 的 LLVM 里自带了）
- 不深入 Dialect 定义（留给 MLIR-L01）
- 重点在"对照"——让读者觉得"这不就是我学过的 LLVM 换了名字吗"

### 2.3 01-Toy-Tutorial导读.md（官方教程带读，新写）

**目标**：读完这篇 + 跟着链接跑完官方教程 Ch1-3

**大纲**：

```
1. Toy 语言简介（3 行）
   - 极简张量计算语言，AI 编译器的微缩模型

2. 环境确认（2 行）
   - mlir-opt 在 PATH 里吗？（brew install llvm 已自带）

3. 三章带读（核心）
   Ch1: 定义 AST → 见 MLIR-L01 对应章节
   Ch2: 定义 Toy Dialect → 见 MLIR-L01 对应章节
   Ch3: 写 Pass 消除冗余 transpose → 见 MLIR-L02 对应章节

   每章配：
   - 这一段在干什么（一句话）
   - 和 LLVM 对应的概念
   - 跑通命令
   - 预期输出

4. 和 HelloPass 的对比
   - Toy Ch3 的 Pass = HelloPass 的 MLIR 版本
   - 都是"遍历 IR → 做点什么"

5. 三章学完后自检
```

**关键设计决策**：
- 不是翻译官方教程，是"带读"——告诉你每章在干什么、为什么要学
- 每章末尾指向 MLIR-L01/L02 的对应章节看详细代码
- 预计 60 分钟（30 分钟读这篇 + 30 分钟跑代码）

### 2.4 02-ascendnpu-ir快速上手.md（项目上手，新写）

**目标**：从 Toy Tutorial 过渡到真实项目

**大纲**：

```
1. ascendnpu-ir 是什么？
   - Ascend NPU 编译器的 MLIR 实现
   - 对比 Toy Tutorial：规模/真实度/Dialect 数量

2. 项目结构速览
   - bishengir/include/ (Dialect 定义)
   - bishengir/lib/Conversion/ (Lowering Pass)
   - bishengir/test/ (测试用例)
   - 给出文件树，标注"先看哪些"

3. 核心 Lowering 路径（一张图）
   linalg.generic → husion.elemwise_binary → hivm.vadd

4. 在哪里继续深入？
   - → MLIR-L00 有完整 5 步 roadmap
   - → MLIR-L03 有手写 Pass 教程
   - → MLIR-L08 有一键运行脚本

5. 验证清单
```

### 2.5 旧文档包装（MLIR-L00~L08）

每篇旧文档**不修改内容**，只在开头统一加：

```markdown
> 📍 本文档属于 Phase 3 MLIR 学习 | [返回入口](./README.md)
> 前置：建议先完成 [00-从LLVM到MLIR](./00-从LLVM到MLIR.md) 和 [01-Toy-Tutorial导读](./01-Toy-Tutorial导读.md)
> 预估时间：XX 分钟
```

并在末尾加：
```markdown
> 📖 遇到不认识的术语？→ [术语表](../glossary.md)
```

**改动量**：每篇 +5 行，9 篇共 +45 行。

### 2.6 文件名重命名

| 旧名 | 新名 | 原因 |
|------|------|------|
| MLIR-L00-速通与AscendNPU-IR实战 | 03-速通与AscendNPU-IR实战 | 延续 00/01/02 编号 |
| MLIR-L01-ToyTutorial速通-Ch1-Ch2 | 04-ToyTutorial速通-Ch1-Ch2 | 同上 |
| MLIR-L02... | 05... | ... |
| ... | ... | ... |

> ⚠️ 重命名需谨慎：旧文档内部可能有互相引用（`[MLIR-L01]`）。先扫描所有交叉引用再决定是否改名。如果交叉引用多，宁可保持原名。

---

## 3. Phase 4 详细设计

### 架构

```
docs/ascend/
├── README.md                    ← 新写：阶段入口
├── 00-Ascend-NPU硬件概述.md     ← 新写
├── 01-husion-hivm-Dialect详解.md ← 新写
├── 02-一个Ascend-Pass详解.md    ← 新写
└── 03-构建与调试指南.md         ← 新写
```

Phase 4 全是新写，因为仓库里没有 Ascend NPU 后端的教程文档。

### 3.1 00-Ascend-NPU硬件概述.md

**目标**：理解 NPU 怎么工作的，不深入硬件细节

**大纲**：

```
1. NPU vs GPU vs CPU（对比表）
   - CPU: 厨师长（什么都会，但慢）
   - GPU: 100个帮厨（同时切菜）
   - NPU: 切菜机（只切菜，极快）

2. Da Vinci 架构
   - Cube Unit（矩阵乘法 16×16）
   - Vector Unit（向量运算）
   - Scalar Unit（标量控制）
   - 三级缓存：L1 → L2 → HBM
   - 厨房类比图

3. 编译器要做什么？
   - 算子融合：多个操作合并为一个
   - 内存管理：什么时候搬数据
   - 指令映射：高层 matmul → Cube Unit 指令

4. 和 CUDA 的对比
   | CUDA | Ascend |
   | NVCC→PTX→SASS | MLIR→husion→hivm |
   | 私有不通用 | MLIR 通用技能 |

5. 验证清单
```

### 3.2 01-husion-hivm-Dialect详解.md

**目标**：理解两个核心 Dialect 的设计和 Lowering 流程

**大纲**：

```
1. 为什么需要两层 Dialect？
   - 厨房类比：linalg=菜谱, husion=备菜计划, hivm=厨房操作

2. husion Dialect — 昇腾融合 IR
   - 核心操作：elemwise_binary, matmul, relu, load/store
   - 融合示例：两个 linalg.generic → 一个 husion.elemwise_binary
   - 为什么融合减少 50% 数据搬运

3. hivm Dialect — 昇腾虚拟指令
   - 核心操作：vadd, vmul, vrelu, mmad, load, store, barrier
   - 从 husion 到 hivm 的 Lowering 示例

4. 完整 Lowering 流程图
   linalg → husion → hivm → 可执行
   每一步的职责和输出

5. 验证清单
```

### 3.3 02-一个Ascend-Pass详解.md

**目标**：逐层拆解一个真实的 MLIR Pass

**大纲**：

```
1. 选 ConvertLinalgToHusion 为例
   - 为什么选这个：最简单、最直观、和 HelloPass 思路一致

2. Pass 的三部分（对应三个文件）
   ① .td 文件：Dialect 和 Operation 定义
   ② .cpp 文件：匹配模式 + 替换逻辑
   ③ 注册代码：怎么加载到 opt

3. 逐行解读 matchAndRewrite
   Step 1: 检查是不是逐元素运算 → isElementwise()
   Step 2: 提取操作类型 → getOpType()
   Step 3: 创建新 Operation → rewriter.create<>()
   Step 4: 替换 → rewriter.replaceOp()

4. 看测试用例
   - 输入 .mlir（linalg 版本，15 行）
   - 输出 .mlir（husion 版本，1 行）
   - 对比理解 Pass 做了什么

5. 和 HelloPass 的对照表

6. 验证清单（含动手：找另一个 Pass 做同样的拆解）
```

### 3.4 03-构建与调试指南.md

**目标**：在自己机器上跑通 ascendnpu-ir

**大纲**：

```
1. 前置依赖
   - LLVM ≥ 18（已有）
   - CMake + Ninja（已有）
   - 30GB 磁盘空间（LLVM 依赖）

2. 构建步骤
   - git clone AscendNPU-IR
   - cmake 配置（LLVM_DIR + MLIR_DIR）
   - ninja 构建（首次 10-30 分钟）

3. 运行测试
   - ninja check-ascendnpu-ir
   - 单文件测试命令

4. 调试方法
   - 方法1：加 llvm::errs() 打印
   - 方法2：--mlir-print-ir-before-all
   - 方法3：逐步 Lowering，每次看中间结果

5. 常见坑（5 个问题 + 解决）

6. 下一步学习建议
```

---

## 4. 衔接设计

### Phase 2 → Phase 3

| Phase 2 最后一篇 | 引导 |
|-----------------|------|
| `docs/llvm/03-LLVM工具箱速览.md` | 末尾已有"→ Phase 3: MLIR 入门（计划中）"，需改为"→ Phase 3: MLIR 入门" |

### Phase 3 → Phase 4

| Phase 3 最后一篇 | 引导 |
|-----------------|------|
| `MLIR-L08` 或新 `02-ascendnpu-ir快速上手.md` | 末尾加"→ Phase 4: Ascend NPU 后端" |

### 全局导航

```
README.md → quickstart.md → primer/ → llvm/ → mlir/ → ascend/
                                    ↑ 每篇文档底部有 "📖 术语表" + "下一章→"
```

---

## 5. 文件产出汇总

### Phase 3

| # | 文件 | 类型 | 预估行数 |
|---|------|------|---------|
| 1 | `docs/mlir/README.md` | 新写 | ~40 |
| 2 | `docs/mlir/00-从LLVM到MLIR.md` | 新写 | ~200 |
| 3 | `docs/mlir/01-Toy-Tutorial导读.md` | 新写 | ~180 |
| 4 | `docs/mlir/02-ascendnpu-ir快速上手.md` | 新写 | ~180 |
| 5-13 | `docs/mlir/MLIR-L00~L08` | 包装 | +45（9篇×5行） |

### Phase 4

| # | 文件 | 类型 | 预估行数 |
|---|------|------|---------|
| 14 | `docs/ascend/README.md` | 新写 | ~40 |
| 15 | `docs/ascend/00-Ascend-NPU硬件概述.md` | 新写 | ~150 |
| 16 | `docs/ascend/01-husion-hivm-Dialect详解.md` | 新写 | ~250 |
| 17 | `docs/ascend/02-一个Ascend-Pass详解.md` | 新写 | ~250 |
| 18 | `docs/ascend/03-构建与调试指南.md` | 新写 | ~120 |

### 全局更新

| # | 文件 | 修改 |
|---|------|------|
| 19 | `docs/quickstart.md` | "之后学什么"表：Phase 3/4 改为 ✅ |
| 20 | `README.md` | Phase 3/4 状态更新 + 链接 |
| 21 | `SUMMARY.md` | 阶段三/四学习目录改为实际文件 |
| 22 | `docs/llvm/03-LLVM工具箱速览.md` | "计划中" → 实际链接 |

**总计**：22 个文件操作（4 新写桥接 + 9 旧文档包装 + 4 新写 Phase 4 + 5 全局更新），≈1600 行新写 + 45 行包装。

---

## 6. 验证设计

整个 Phase 3+4 做完后，一个读者的完整路径：

```
Phase 2 结束 → 打开 mlir/README.md
           → 读 00-从LLVM到MLIR（15min）→ 能做 LLVM↔MLIR 对照
           → 读 01-Toy-Tutorial导读（30min）+ 跑官方代码（30min）
           → 读 02-ascendnpu-ir快速上手（20min）→ 知道项目长啥样
           → 选读 MLIR-L00~L08（选感兴趣的深入）
           → 进入 Phase 4：ascend/README.md
           → 读 00-NPU硬件概述（10min）
           → 读 01-Dialect详解（20min）
           → 读 02-Pass详解（25min）
           → 读 03-构建指南（15min）+ 自己构建（30min+）
           → 🎉 完成全部学习路径
```

---

## 7. 设计决策记录

| 决策 | 理由 |
|------|------|
| 旧文档不改内容只包装 | 尊重原作者劳动，降低冲突风险 |
| 新写 3 篇桥接而不是修旧文档 | 桥接是"翻译层"，旧文档是"内容层"，各司其职 |
| Phase 4 全新写 | 仓库里没有 Ascend NPU 后端教程 |
| 保持 Phase 1/2 风格 | 统一的比喻/对照表/验证清单让学习体验一致 |
| 文件名延续 00/01/02 编号 | 和 primer、llvm 目录风格统一 |
