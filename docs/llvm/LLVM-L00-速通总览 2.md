---
tags:
  - 工程
---

> **项目关联**: AA-Triton-Ascend 昇腾后端开发
> Triton 语言前端 → LLVM IR → 自定义后端 Pass → Ascend NPU 指令
> LLVM IR 和 Pass 开发知识直接用于 Triton 后端的代码生成和优化。

---

# LLVM 速通总览

基于 [llvm-ir-tutorial](https://github.com/Evian-Zhang/llvm-ir-tutorial) + [llvm-tutor](https://github.com/banach-space/llvm-tutor)

## 学习路线

| # | 主题 | 核心概念 |
|---|------|---------|
| 1 | LLVM 架构 + Hello World | 三段式架构、`.ll` 文件结构、SSA 形式 |
| 2 | 类型系统 + GEP | `iN`、类型转换、GEP 地址运算 |
| 3 | 控制流 + φ 节点 | `br`、φ 指令、for 循环模式的 φ |
| 4 | 内置函数 + 属性 | `llvm.memcpy`、函数属性组、元数据 |
| 5 | LLVM Pass 开发 | `opt -load-pass-plugin`、out-of-tree Pass |
| 6 | 自写 Pass（BBCounter） | 函数→基本块遍历、统计 Pass |

## 关键命令

```bash
clang file.ll -o file          # 编译 .ll->可执行
clang -S -emit-llvm test.c     # C->LLVM IR
opt -load-pass-plugin lib.dylib -passes="name" file.ll  # 跑 Pass
llvm-config --version          # 查看 LLVM 版本
```

## 学习资源

| 资源 | 链接 |
|------|------|
| llvm-tutor | https://github.com/banach-space/llvm-tutor (3.4k⭐) |
| LLVM IR 入门指南 | https://github.com/Evian-Zhang/llvm-ir-tutorial (1.5k⭐) |
| LLVM 语言参考 | https://llvm.org/docs/LangRef.html |
| LLVM 编程手册 | https://llvm.org/docs/ProgrammersManual.html |
