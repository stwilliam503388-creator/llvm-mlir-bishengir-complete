---
tags:
  - 工程
---

# LLVM 控制流与 φ 节点

## 分支指令

LLVM 只有一种 `br`，两种形态：

```llvm
br i1 %cond, label %true, label %false  ; 条件分支
br label %target                         ; 无条件跳转
```

比较：

```llvm
%cmp = icmp sgt i32 %a, %b  ; signed greater than
%fcmp = fcmp oeq float %x, %y ; 浮点相等
```

## φ（Phi）节点 — 为什么需要它

SSA 规则：每个变量只赋值一次。`if..else` 汇合时，值来自哪条路径？

```llvm
; C: if (cond) c=10 else c=20
then:
    %c1 = add i32 0, 10
    br label %merge
else:
    %c2 = add i32 0, 20
    br label %merge
merge:
    %c = phi i32 [%c1, %then], [%c2, %else]  ; φ 选择
    ret i32 %c
```

**φ = "从哪条路来，取哪个值"**

语法：`phi i32 [值1, 来源块1], [值2, 来源块2]`

**重要**：φ 不是运算指令——机器码中消失（被寄存器分配替代）。

## For 循环中的 φ

```llvm
loop_entry:
    %i   = phi i32 [ 0, %0 ],          [ %i_next,   %loop_body ]
    %sum = phi i32 [ 0, %0 ],          [ %sum_next, %loop_body ]
                    ↑ 第一次进          ↑ 后续迭代（自引用）
```

完整例子（0+1+...+9=45）：

```llvm
define i32 @main() {
    br label %loop_entry
loop_entry:
    %i = phi i32 [0, %0], [%i_next, %loop_body]
    %sum = phi i32 [0, %0], [%sum_next, %loop_body]
    %cond = icmp slt i32 %i, 10
    br i1 %cond, label %loop_body, label %exit
loop_body:
    %sum_next = add i32 %sum, %i
    %i_next = add i32 %i, 1
    br label %loop_entry
exit:
    ret i32 %sum
}
```

## 函数声明与调用

```llvm
declare i32 @printf(ptr, ...)                ; 声明外部函数
@.str = private constant [4 x i8] c"%d\\0A\\00"  ; 字符串常量
call i32 (ptr, ...) @printf(ptr @.str, i32 %sum) ; 调用
```

| 关键字 | 含义 |
|--------|------|
| `define` | 定义函数（有函数体） |
| `declare` | 声明函数（只有签名） |
| `private` | 链接类型：不出符号表 |
| `c"...\\00"` | C 风格字符串 |
