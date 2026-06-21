# 编译器零基础入门（Primer）

> 写给 AI 工程师的编译器概念速成。
> 不需要任何编译器经验。会用 Python 就能读。

> 💡 **遇到不认识的术语？** 查 `docs/reference/技术术语速查手册.md`，每条术语含"一句话"+"类比"+"为什么重要"。

## 阅读顺序

```
00 ← 从这里开始 (8min) — 编译器为什么存在？三步工作法？
  └─→ 01 (8min) — AST 和 IR 是什么？长得什么样？
       └─→ 02 (8min) — Pass 和 Lowering 怎么把代码变成机器码？
            └─→ 03 (5min) — 串联到本项目：从 Triton 到 Ascend NPU
```

**总共约 30 分钟。**

读完这 4 篇后，你就能：
- 知道编译器是干什么的
- 区分 AST 和 IR
- 理解 SSA 的"每个变量只赋值一次"
- 知道 Pass 分为"分析"和"转换"两种
- 理解 Lowering 为什么要把 1 行变成 74 行
- 看懂本项目的学习路径图

## 每个概念对应的项目位置

| 概念 | 对应文件 |
|------|---------|
| AST | `projects/toy-mini/toymini.cpp` → `struct NumberExpr` |
| IR | `projects/standalone-mlir/test/example.mlir` |
| SSA | `projects/standalone-mlir/test/example.mlir` → `%0, %1, %2` |
| 分析 Pass | `projects/bishengir-op-counter/BishengirOpCounter.cpp` |
| 转换 Pass | `projects/bishengir-op-counter/BishengirPeelTranspose.cpp` |
| Lowering | `projects/bishengir-demo/variants/compare.sh` |
| Dialect | `projects/standalone-mlir/include/standalone/StandaloneOps.td` |

## 读完 Primer 之后

进入 `docs/` 目录，从 LLVM 笔记（L01）开始。遇到不理解的概念：
1. 先查 `docs/reference/技术术语速查手册.md`
2. 找不到再回 Primer 搜索

## 术语对照

| 英文 | 中文 | 一句话 | 详见 |
|------|------|--------|------|
| AST | 抽象语法树 | 代码的"语法树"，告诉你代码结构 | Primer 01 |
| IR | 中间表示 | 编译器的"通用语言" | Primer 01 |
| SSA | 静态单赋值 | 每个变量只赋值一次 | Primer 01 |
| Pass | 遍/趟 | 遍历 IR 做一次修改 | Primer 02 |
| Lowering | 降级 | 从高级 IR 到低级 IR | Primer 02 |
| Dialect | 方言 | MLIR 的"主题词汇表" | Primer 02 |
