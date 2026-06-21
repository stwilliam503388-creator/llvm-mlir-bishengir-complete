---
type: plan
status: approved
project: llvm-mlir-bishengir-complete
created: 2026-06-21
---

# 零基础编译器入门：扩充方案

## 目标读者画像

"我是做 AI 应用的，会用 PyTorch/Triton 写模型，但从没学过编译器。
看到 AST、IR、SSA、Pass 这些词完全懵。我需要有人用我能听懂的话解释：
编译器到底在干嘛？这些概念跟我写代码有什么关系？"

## 新增文档

在 `docs/primer/` 下写 4 篇文档（之前计划的 2 篇 → 扩充为 4 篇）。

### 文档 00: `编译器是什么 —— 写给 AI 工程师的零基础入门`

> 阅读时间: 8 分钟
> 篇幅: ~300 行
> 目标: 让读者理解编译器的存在意义、基本工作流程、与他们的日常工作的关系。

```
├── 1. 一条代码的执行之路
│   ├── Python: 解释器逐行读 → 边读边执行
│   ├── C++:  源代码 → 编译 → 二进制 → 执行
│   ├── Triton: Python 写 → 编译成 GPU/NPU 指令 → 执行
│   └── 关键问题: 为什么需要编译？—— 人和机器说的不是同一种语言
│
├── 2. 编译器的三步工作法
│   ├── 前端 (Frontend):   读懂你的代码 → 建 AST
│   │   └── 类比: 英语老师分析句子结构
│   ├── 中端 (Middle-end): 优化代码 → IR 转换
│   │   └── 类比: 把"3+5+2"改成"10"——结果一样，计算更快
│   ├── 后端 (Backend):    生成机器码 → 执行
│   │   └── 类比: 把中文作文翻译成英文演讲
│   └── 为什么分三步？—— 换手机(Arm→x86)只需换后端
│
├── 3. 你在本项目中会看到什么
│   ├── LLVM IR 基础   = 中端 IR 的一种通用形式
│   ├── MLIR dialect   = 多阶段 IR，每层解决不同问题
│   ├── AscendNPU-IR   = 从通用 IR 到 Ascend NPU 专用 IR
│   └── 贯穿全项目: AST → IR1 → IR2 → IR3 → 机器码
│
└── 4. 读完本节后你能回答
    ├── 编译器和解释器有什么区别？
    ├── 为什么 Triton 需要 "编译" 而 Python 不用？
    └── LLVM、MLIR、AscendNPU-IR 都是 IR，为什么要有好几个？
```

### 文档 01: `AST 与 IR —— 代码的两种中间形态`

> 阅读时间: 8 分钟
> 篇幅: ~300 行
> 目标: 让读者理解 AST 和 IR 是什么、长什么样、有什么用。

```
├── 1. AST — 代码的语法树
│   ├── 问题: 计算机怎么"理解" `a + b * 3` 不是字符串？
│   ├── 答案: 拆成一棵树
│   │        ┌─── + ───┐
│   │        a      ┌─── * ───┐
│   │               b         3
│   ├── 术语: 根节点、子节点、叶子节点
│   ├── 在项目中的样子: toymini.cpp 的 AST.h
│   │   ├── NumberExpr     — 数字节点 (叶子)
│   │   ├── BinaryExpr     — 二元运算节点 (有左右子树)
│   │   └── FunctionAST    — 函数定义节点 (有参数列表和函数体)
│   └── AST 的特点: 保留了全部的语法信息（括号、优先级都体现在树结构里）
│
├── 2. IR — 中间表示
│   ├── 问题: AST 包含太多语法细节，不利于优化
│   ├── 答案: 拍平成线性指令序列
│   ├── 三地址码 (Three-Address Code):
│   │   AST: a + b * 3
│   │   IR:  %1 = mul b, 3     # 每个指令最多一个运算符
│   │        %2 = add a, %1    # 结果存在临时变量里
│   ├── SSA (Static Single Assignment):
│   │   规则: 每个变量只赋值一次
│   │   好处: 一眼能看出数据依赖关系
│   │   LLVM IR: %1 = mul i32 %b, 3
│   │            %2 = add i32 %a, %1
│   ├── 在项目中的样子:
│   │   └── LLVM 笔记 L01: 能看懂 .ll 文件
│   │   └── standalone-mlir: 能看到自己 dialect 的 IR 输出
│   └── IR 的特点: 去掉了语法糖，保留了数据流和控制流
│
├── 3. AST vs IR 对比
│   ├── AST 像"建筑设计图"     — 保留了门、窗、墙壁的位置
│   ├── IR  像"施工指令序列"   — 先砌墙、再安门、再装窗
│   ├── AST 适合语法分析        (告诉我写了什么)
│   └── IR  适合优化和执行      (告诉我怎么做)
│
└── 4. 读完本节后你能回答
    ├── `1 + 2 * 3` 在 AST 中是树还是列表？
    ├── SSA 的"每个变量只赋值一次"是什么意思？
    └── AST.h 里的 NumberExpr 和 BinaryExpr 是什么关系？
```

### 文档 02: `Pass 与 Lowering —— IR 是怎么一步步变成机器码的`

> 阅读时间: 8 分钟
> 篇幅: ~250 行
> 目标: 理解 Pass 和 Lowering 这两个贯穿全项目的核心概念。

```
├── 1. Pass — IR 的"修改器"
│   ├── 定义: 一个 Pass = 对着 IR 做一次遍历 + 一次修改
│   ├── 两种 Pass:
│   │   ├── 分析 Pass: 只看不改 (统计用了多少 add、检测死代码)
│   │   └── 转换 Pass: 边看边改 (把 add 替换成 mul+add、删除无用变量)
│   ├── 类比: 质检员检查流水线
│   │   ├── 分析 Pass = 质检员数一数今天产了多少件
│   │   └── 转换 Pass = 质检员发现有瑕疵，直接修复
│   ├── 在项目中的样子:
│   │   └── ascendnpu-ir-op-counter/BishengirOpCounter.cpp
│   │       - 遍历所有 op → 计数 → 打印 (分析 Pass)
│   │   └── ascendnpu-ir-op-counter/BishengirPeelTranspose.cpp
│   │       - 检测冗余 transpose → 删除 → 替换 (转换 Pass)
│   └── Pass 管线: 多个 Pass 按顺序执行
│
├── 2. Lowering — 从高级到低级
│   ├── 定义: 把"易写但难执行"的 IR 变成"难写但易执行"的 IR
│   ├── 一组类比:
│   │   ├── linalg.matmul (一行)          = "做一桌满汉全席"
│   │   ├── affine.for × 3 (18 行)        = "先买菜、再洗菜、再炒菜"
│   │   └── llvm IR (74 行)               = "每一种食材切几刀、炒几分钟"
│   ├── 为什么不能一步到位？
│   │   └── 中间层可以做优化: 比如发现两个循环可以合并
│   ├── 在项目中的样子:
│   │   └── ascendnpu-ir-demo: 三阶段降级
│   │       Linalg → affine → scf → LLVM (74× 膨胀)
│   │   └── AscendNPU-IR 实际: 
│   │       Linalg → HFusion → HIVM → NPU (保持 1 行，硬件指令)
│   └── 关键认识: 膨胀不是 bug——展开得越细，优化空间越大
│
├── 3. Dialect — MLIR 的多层 IR
│   ├── 传统编译器: 1 个 IR 走到底 (LLVM)
│   │   └── 问题: 硬件差异大时，一个 IR 不够灵活
│   ├── MLIR: N 个 IR (dialect) 接力
│   │   ├── Toy Dialect: 适合写 Toy 程序
│   │   ├── Linalg Dialect: 适合表达矩阵运算
│   │   ├── Affine Dialect: 适合表达循环
│   │   └── LLVM Dialect: 适合生成机器码
│   ├── 类比: 翻译团队
│   │   ├── 英文 → 中文全程一个人翻 = 传统 1 个 IR
│   │   └── 英文→日文→中文三个人接力 = MLIR 多个 dialect
│   │       每个人只翻自己最擅长的段落
│   └── 在项目中的样子:
│       └── standalone-mlir: 自定义 standalone dialect
│       └── triton: TT dialect → TritonGPU dialect → LLVM
│
└── 4. 读完本节后你能回答
    ├── ascendnpu-ir-demo 的 74 行 LLVM 是"问题"还是"过程"？
    ├── 为什么 MLIR 不用一个 IR 而用多个 dialect？
    └── 分析 Pass 和转换 Pass 的区别是什么？
```

### 文档 03: `从 Triton 到 Ascend —— 贯穿全项目的完整路径`

> 阅读时间: 5 分钟
> 篇幅: ~150 行
> 目标: 把前面 3 篇的知识串联到一条具体的执行路径上。

```
├── 1. 我们最终要实现什么？
│   ├── 写一段 Triton Python 代码
│   ├── 让它跑到 Ascend NPU 上
│   └── 中间发生了什么？
│
├── 2. 全路径图（带每步对应的概念）
│
│   Triton Python  (你的代码)
│       │
│       ▼  [AST 构建]  ← 文档 01
│   Python AST
│       │
│       ▼  [IR 生成]  ← 文档 01
│   Triton IR (tt dialect)   ← 高级 IR
│       │
│       ▼  [Pass + Lowering]  ← 文档 02
│   TritonGPU IR              ← 中级 IR (加上了内存布局)
│       │
│       ▼  [Pass + Lowering]  ← 文档 02
│   LLVM IR / AIR             ← 低级 IR
│       │
│       ▼  [Backend]
│   Ascend NPU 指令           ← 机器码
│
├── 3. 本项目覆盖了哪些部分
│   ├── docs/llvm/L01-L06:  理解 LLVM IR
│   ├── docs/mlir/L00-L02:  理解 MLIR dialect + Pass
│   ├── standalone-mlir:    自己定义一个 dialect + Pass
│   ├── ascendnpu-ir-demo:     模拟降级过程
│   ├── ascendnpu-ir-op-counter:写自定义 Pass
│   ├── toy-mini:           自己写一个 AST 解析器
│   └── docs/mlir/L06-L07:  Triton 怎么接 AscendNPU-IR
│
└── 4. 读完本节后你能回答
    ├── 整个项目学完后我能做什么？
    └── 我现在在哪一步？
```

## 整合 README

| 改动位置 | 修改内容 |
|---------|---------|
| README 前置知识表 | "❌ 编译器经验" → "🟡 编译器经验（从零开始也不怕 → `docs/primer/`）" |
| README 快速入门 | 新增一行：`# 0. 零基础？先读 docs/primer/` |
| README 学习路径 | Stage 0 前新增 "Stage -1: 编译器零基础入门（可选）" |
| README 新增引用块 | 第一章末尾：**没有编译器基础？** 指向 primer 01 |

## 文件清单

```
docs/primer/
├── README.md                      — 本目录说明 + 阅读顺序
├── 00-编译器是什么.md              — 编译器为什么存在
├── 01-AST与IR.md                   — 两种中间形态
├── 02-Pass与Lowering.md            — 修改器与降级
└── 03-从Triton到Ascend.md          — 全路径串联
```
