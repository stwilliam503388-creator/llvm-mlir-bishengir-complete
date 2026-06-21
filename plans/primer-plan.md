> ✅ **此计划已执行完毕**。结果见 `docs/primer/`（4 篇文档）。

# 补充计划：面向非编译器背景读者的入门指引

## 问题

现有 README 在"前置知识"中要求"编译器直觉，了解 AST、IR 概念"，
但没有提供面向零基础读者的快速入门材料。ML/AI 工程师通常熟悉 PyTorch/Triton 使用，
但对编译器内部的 AST、IR、SSA、Pass 等概念陌生。

## 方案

在 `docs/primer/` 下新增 2 篇文档，用 AI 工程师熟悉的类比解释编译器概念。

### 文档一: `docs/primer/01-从Python到NPU-编译器的基本概念.md`

目标: 让读者建立"编译器是做什么的"直觉。
面向: 写过 Python/PyTorch/Triton 但没写过编译器的读者。
篇幅: ~3-5 分钟阅读。

```
内容大纲:
├── 1. 你到底在学什么？
│   ├── 一条 Triton 代码到 NPU 执行的路径
│   └── 编译器 = 翻译官 + 优化师
├── 2. AST — 代码的语法树
│   ├── 类比: 语文课的句子成分分析
│   ├── Python 例: `a + b * 3` → AST 树
│   └── 在我们的项目中: Toy Mini 的 parseExpr() 就是建 AST
├── 3. IR — 中间表示
│   ├── 类比: 英文→中文翻译时的"中间笔记"
│   ├── 三地址码、SSA 形式
│   └── 在我们的项目中: LLVM IR (.ll)、MLIR (tensor/memref)
├── 4. Pass — IR 的转换器
│   ├── 类比: 流水线上的工作站
│   ├── 分析 Pass (只看不改) vs 转换 Pass (改了再写)
│   └── 在我们的项目中: bishengir-op-counter 的两种 Pass
├── 5. Lowering — 从抽象到具体
│   ├── 类比: 建筑设计图 → 施工图 → 钢筋清单
│   ├── 多层 IR 的核心价值
│   └── 在我们的项目中: bishengir-demo 的三阶段降级
└── 6. 接下来该读什么
    └── 指向 LLVM L01 / MLIR L00
```

### 文档二: `docs/primer/02-SSA与MLIRDialect-快速理解.md`

目标: 从 SSA 到 MLIR dialect 的快速衔接。
面向: 读完 Primer 01 后，准备进入 LLVM 笔记的读者。
篇幅: ~3 分钟阅读。

```
内容大纲:
├── 1. SSA 一句话
│   ├── 每个变量只赋值一次
│   ├── 好处: 数据流 = 控制流，依赖一目了然
│   └── φ 节点: 分支汇合时的选择器
├── 2. MLIR 的多层 IR 是什么
│   ├── 传统编译器: 1 个 IR 干所有事 (LLVM)
│   ├── MLIR 方案: N 个 IR 各管一段
│   └── 类比: 一个翻译不够就加两个
├── 3. dialect / operation / region
│   ├── dialect = 方言 (一群人用的词汇)
│   ├── operation = 词汇 (load/add/store)
│   ├── region = 语境 (操作的作用域)
│   └── 在我们的项目中: standalone Dialect 的 6 个 op
├── 4. 回到 README 的学习路径
│   └── 你现在有基础了，从 Stage 0 开始
```

### 修改 README

| 改动 | 位置 | 内容 |
|------|------|------|
| 前置知识表 | 第一章最后一节 | "❌ 编译器经验" → "🟡 编译器经验 (有 Primer 可补)" |
| 学习路径前 | 第三章开头 | 新增"不需要编译器经验，从 Primer 01 开始" |
| 新增引用 | 第一章末尾 | "没有编译器基础？从 docs/primer/01-... 开始" |

### 文件清单

```
docs/primer/
├── README.md          — 本目录说明
├── 01-从Python到NPU-编译器的基本概念.md    — AST/IR/Pass/Lowering 讲解
└── 02-SSA与MLIRDialect-快速理解.md        — SSA → dialect → region 衔接
```

### 实施步骤

1. 写 `01-从Python到NPU-编译器的基本概念.md` (~200 行)
2. 写 `02-SSA与MLIRDialect-快速理解.md` (~100 行)
3. 写 `docs/primer/README.md`
4. 更新根 README 前置知识 + 学习路径部分
5. git add + commit + push

---

这个方案可以吗？需要调整内容侧重点或篇幅吗？
