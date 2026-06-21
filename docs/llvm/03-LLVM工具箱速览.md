# 03 - LLVM 工具箱速览

> 目标：知道 LLVM 有哪些工具，各干什么，需要时回来查
> 前置：[02 - 第一个 LLVM Pass](./02-第一个LLVM-Pass.md)
> 预估时间：10 分钟

---

## 不用全学会

LLVM 全家桶有几十个工具。日常用到的就 5 个。

---

## 核心工具速查

| 工具 | 一句话 | 最常用命令 |
|------|--------|-----------|
| **clang** | C/C++ → IR | `clang -S -emit-llvm hello.c -o hello.ll` |
| **opt** | 在 IR 上跑 Pass | `opt -load-pass-plugin ./libMyPass.dylib --passes="mypass" input.ll` |
| **llc** | IR → 汇编 | `llc hello.ll -o hello.s` |
| **lli** | 直接运行 IR | `lli hello.ll` |
| **llvm-dis** | .bc → .ll | `llvm-dis hello.bc -o hello.ll` |

---

## 常用命令速查

```bash
# C → LLVM IR（文本格式）
clang -S -emit-llvm hello.c -o hello.ll

# C → LLVM IR（二进制格式，.bc）
clang -c -emit-llvm hello.c -o hello.bc

# 优化 IR
opt -S hello.ll -o optimized.ll

# 加载你的 Pass
opt -load-pass-plugin ./libMyPass.dylib --passes="mypass" input.ll -S

# IR → 汇编
llc hello.ll -o hello.s

# 直接运行 IR（不编译）
lli hello.ll

# 二进制 → 文本
llvm-dis hello.bc -o hello.ll
```

---

## 文件格式

| 后缀 | 格式 | 能直接读吗 |
|------|------|-----------|
| .ll | LLVM IR 文本 | ✅ 能 |
| .bc | LLVM IR 二进制 | ❌ 需要用 llvm-dis 转成 .ll |
| .s | 汇编文本 | 勉强 |
| .o | 目标文件 | ❌ |

---

## 不需要记

这页是用来查的，不是用来背的。写 Pass 写到一半忘了命令，回来翻就行。

---

> 📖 遇到不认识的术语？→ [术语表](../glossary.md)

> **下一步**：[Phase 3: MLIR 入门](../../plans/phase3-4-implementation-plan.md)（计划中）
