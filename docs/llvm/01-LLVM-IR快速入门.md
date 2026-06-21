# 01 - LLVM IR 快速入门

> 目标：能读懂 LLVM IR，会用 clang 自己生成 IR
> 前置：[00 - 环境搭建](./00-环境搭建.md)
> 预估时间：30 分钟

---

## 1. 回顾：IR 在编译流水线中的位置

在 [Primer 01 - AST 与 IR](../primer/01-AST与IR.md) 里我们比喻 IR 是"蓝图纸"——
AST 是草稿，IR 是可执行的精确蓝图。

LLVM IR 就是编译器世界里通用的蓝图纸语言。所有语言的编译器前端（C、Rust、Swift…）都翻译成同一种 LLVM IR，
然后后端统一处理。

```
C 源码  →  clang 前端  →  LLVM IR  →  opt 优化  →  llc  →  机器码
         (parse)        (.ll文件)   (你的Pass在此)
```

---

## 2. 第一眼：一个加法函数

### C 代码

```c
int add(int a, int b) {
    return a + b;
}
```

### 生成 LLVM IR

```bash
clang -S -emit-llvm add.c -o add.ll
cat add.ll
```

你会看到类似这样的输出（简化版）：

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

### 逐行解读

| 行 | 含义 | 人话 |
|----|------|------|
| `define i32 @add(i32 %a, i32 %b)` | 定义一个函数 add，返回 i32，接受两个 i32 参数 | "函数 add 吃什么吐什么" |
| `{` | 函数体开始 | |
| `entry:` | 基本块标签，叫 entry（入口块） | "这是函数的入口" |
| `%sum = add i32 %a, %b` | i32 类型的加法指令，结果存在虚拟寄存器 %sum | "把 a 和 b 加到一起叫 sum" |
| `ret i32 %sum` | 返回 %sum | "把 sum 交给调用者" |
| `}` | 函数体结束 | |

---

## 3. 三个核心概念

### 3.1 SSA（Static Single Assignment，静态单赋值）

**规则：每个变量只赋值一次，永不修改。**

```
❌ 错误（不是 SSA）:
x = 1
x = x + 1     // x 被赋值了两次！

✅ 正确（SSA）:
%x1 = 1
%x2 = add %x1, 1   // 新值用新名字
```

**比喻**：流水线上每个工位只加工一次就把工件传给下一个工位。不会在一个工位上反复修改同一个工件。

**为什么编译器喜欢 SSA？** 每次赋值都是独立的"事实"，编译器不需要追踪"x 现在是几"。这大幅简化了分析和优化。

### 3.2 基本块（Basic Block）

**定义：一段连续的指令序列，只有一个人口和一个出口。中间不能有分支。**

**比喻**：高速公路的直线路段 — 从入口进来，从出口出去，中间不能变道或转弯。

```llvm
entry:           ; ← 基本块入口
  %result = ...
  br label %next ; ← 基本块出口（跳转）

next:            ; ← 下一个基本块
  ...
```

**判断方法**：看最后一条指令。如果不是 `br`/`ret`/`switch` 等控制流指令，那后面还能接指令，没结束。

### 3.3 phi 节点

**phi 节点是多路汇合的选择器 — 根据"我是从哪条路进来的"选择不同的值。**

**比喻**：立交桥合流处。你从路 A 来就选 A 的值，从路 B 来就选 B 的值。

```llvm
; 一个 if-else 条件分支的 IR
  br i1 %cond, label %then, label %else

then:
  %val = add i32 1, 1
  br label %merge

else:
  %val2 = sub i32 10, 1
  br label %merge

merge:
  %result = phi i32 [ %val, %then ], [ %val2, %else ]
  ; ↑ 意思是：如果从 %then 块来，用 %val；
  ;            如果从 %else 块来，用 %val2
```

---

## 4. 动手：自己生成 IR

### 练习

写一个 `compare.c`：

```c
int max(int a, int b) {
    if (a > b)
        return a;
    else
        return b;
}
```

```bash
clang -S -emit-llvm compare.c -o compare.ll
cat compare.ll
```

### 预期结果

你应该看到类似这样的 IR（简化版）：

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b        ; 比较 a > b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %return                    ; 跳转到汇合点

if.else:
  br label %return

return:
  %retval = phi i32 [ %a, %if.then ], [ %b, %if.else ]
  ret i32 %retval
}
```

逐行对照看：

| IR 代码 | 对应用 C 代码 |
|---------|-------------|
| `icmp sgt i32 %a, %b` | `a > b` |
| `br i1 %cmp, label %if.then, label %if.else` | `if (a > b)` |
| `phi i32 [ %a, %if.then ], [ %b, %if.else ]` | 来自 then 返回 a，来自 else 返回 b |
| `ret i32 %retval` | `return` |

> ⚠️ 实际输出可能和上面不完全一样（变量名、基本块名可能不同），但核心结构相同：一个 icmp + 两个分支 + 一个 phi + 一个 ret。

**检查清单**：
- [ ] 找到了函数签名 `define i32 @max(i32, i32)`
- [ ] 找到了 `icmp` 指令（比较 a > b）
- [ ] 至少有两个基本块（if.then 和 if.else）
- [ ] 在汇合点（return 块）能找到 `phi` 节点
- [ ] 每个变量（以 `%` 开头）只被赋值一次（SSA）

---

## 5. 小结：学了 IR 能干什么？

1. 看懂编译器中间产物，理解"编译器到底干了什么"
2. 为写 Pass 打基础 — Pass 就是在 IR 上做检查和变换
3. 调试优化问题 — 对比优化前后的 IR 看出优化效果

---

## 下一步

→ [02 - 第一个 LLVM Pass](./02-第一个LLVM-Pass.md) — 手写代码，跑起来

---

## 本节验证清单

完成以下所有项才算学完：

- [ ] 能用 `clang -S -emit-llvm` 生成 .ll 文件
- [ ] 能指出 IR 中的函数签名、基本块、指令
- [ ] 能用一句话解释 SSA：每个变量只赋值一次
- [ ] 能用一句话解释基本块：连续的直线代码，一个入口一个出口
- [ ] 能找到了 phi 节点并解释它的作用

> 📖 遇到不认识的术语？→ [术语表](../glossary.md)
