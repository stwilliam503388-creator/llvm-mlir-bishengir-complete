---
tags:
  - 工程
---

# LLVM 架构与 Hello World

## 为什么有 LLVM？

传统方案：自己发明的语言 → 编译成 C 代码 → 各平台 C 编译器处理。
问题：C 太抽象，无法精细控制底层行为。

LLVM 方案：
```
你的语言源码 → 前端解析 → LLVM IR → [优化] → 机器码
               (你写)           (LLVM 管)
```

## 三段式架构

```
 前端 (Frontend)     中端 (Middle-end)       后端 (Backend)
 Clang/Rustc ──→ LLVM IR ──→ opt 优化 ──→ CodeGen ──→ .s → .o
                    ↑ 承上启下                     ↓
             语言无关的优化层                 平台相关代码生成
```

## LLVM IR 的三种形态

| 形态 | 后缀 | 说明 |
|------|------|------|
| 文本 | `.ll` | 人类可读，手写 |
| 二进制 | `.bc` | 紧凑，内部传递 |
| 内存结构 | — | Pass 操作的对象 |

## 第一个程序

```llvm
; main.ll
define i32 @main() {
    ret i32 42
}
```
```bash
clang main.ll -o main && ./main; echo $?   # 42
```

## 关键语法

| `@add` | `@` = 全局符号（函数名、全局变量） |
| `%a` | `%` = 局部符号（参数、临时变量） |
| `i32` | 类型——LLVM 是强类型的 |
| `add i32 %a, %b` | 指令格式 = 操作码 + 类型 + 操作数 |
| `%result =` | SSA——每个变量只能赋值一次 |

## SSA 形式（Static Single Assignment）

**每个变量只被赋值一次，然后永远不变。** SSA 让优化器可以安全地重排指令——不需要跟踪变量值的变化。

## 从 C 生成 IR

```bash
clang -S -emit-llvm -O0 test.c -o test.ll
```

## 栈上分配

```llvm
%a = alloca i32           ; 在栈上分配空间
store i32 15, ptr %a      ; 写入
%x = load i32, ptr %a     ; 读取
```

## 注意事项

- LLVM 是强类型的（每个值都有类型）
- Apple Clang 可以编 `.ll`；写 Pass 需要完整的 LLVM 开发库
- 对于 Ascend 后端开发：理解 IR 是写自定义 Pass 的前提
