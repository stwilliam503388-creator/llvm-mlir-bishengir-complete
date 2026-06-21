# 术语表

编译器领域术语密集。遇到不认识的词，回到这里查。

> 初次出现的术语在中英对照表中列出，核心概念附一句话解释。

---

## 核心概念

| 术语 | 英文 | 一句话解释 | 在哪学 |
|------|------|-----------|--------|
| 编译器 | Compiler | 把源代码翻译成机器码的程序 | [primer/00](../primer/00-编译器是什么.md) |
| 解释器 | Interpreter | 逐行执行源代码，不生成机器码 | [primer/00](../primer/00-编译器是什么.md) |
| 前端 | Frontend | 编译器的"入口"，解析源码生成 AST | [primer/01](../primer/01-AST与IR.md) |
| 后端 | Backend | 编译器的"出口"，把 IR 变成机器码 | [primer/00](../primer/00-编译器是什么.md) |
| AST | Abstract Syntax Tree | 抽象语法树，源码的树形表示 | [primer/01](../primer/01-AST与IR.md) |
| IR | Intermediate Representation | 中间表示，前端和后端之间的"通用语言" | [primer/01](../primer/01-AST与IR.md) |
| SSA | Static Single Assignment | 静态单赋值：每个变量只赋值一次 | [llvm/01](../llvm/01-LLVM-IR快速入门.md) |
| 基本块 | Basic Block | 连续执行的直线代码，一个入口一个出口 | [llvm/01](../llvm/01-LLVM-IR快速入门.md) |
| phi 节点 | Phi Node | 多路汇合处的选择器，根据来路选值 | [llvm/01](../llvm/01-LLVM-IR快速入门.md) |
| Pass | Pass | 对 IR 做一次检查或变换的"质检员" | [primer/02](../primer/02-Pass与Lowering.md) |
| Lowering | Lowering | 把高级 IR 逐步"降级"到底层 IR 的过程 | [primer/02](../primer/02-Pass与Lowering.md) |

## LLVM 工具

| 工具 | 全称 | 作用 | 在哪学 |
|------|------|------|--------|
| clang | C language | C/C++ 编译器前端 | [llvm/03](../llvm/03-LLVM工具箱速览.md) |
| opt | Optimizer | IR 优化器，加载 Pass 运行 | [llvm/02](../llvm/02-第一个LLVM-Pass.md) |
| llc | LLVM Compiler | IR → 汇编代码 | [llvm/03](../llvm/03-LLVM工具箱速览.md) |
| lli | LLVM Interpreter | 直接运行 IR（不编译） | [llvm/03](../llvm/03-LLVM工具箱速览.md) |
| llvm-dis | LLVM Disassembler | 二进制 IR → 可读文本 | [llvm/03](../llvm/03-LLVM工具箱速览.md) |

## 文件格式

| 后缀 | 含义 | 能直接读吗 |
|------|------|-----------|
| .c / .cpp | C/C++ 源码 | ✅ |
| .ll | LLVM IR 文本格式 | ✅ |
| .bc | LLVM IR 二进制格式 | ❌ 需用 llvm-dis 转 |
| .s | 汇编代码 | 勉强 |
| .o | 目标文件 | ❌ |

## Ascend 相关

| 术语 | 全称 | 一句话解释 |
|------|------|-----------|
| NPU | Neural Processing Unit | 神经网络处理器，华为 Ascend 的芯片类型 |
| CANN | Compute Architecture for Neural Networks | 华为 Ascend 的软件栈 |
| TBE | Tensor Boost Engine | CANN 中的算子开发框架 |
| Ascend C | — | 华为 NPU 的 C++ 扩展编程语言 |
