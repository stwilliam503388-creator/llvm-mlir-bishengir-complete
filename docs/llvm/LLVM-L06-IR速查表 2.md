---
tags:
  - 工程
---

# LLVM IR 速查表

## 常用指令

| 指令 | 语法 | 说明 |
|------|------|------|
| `ret` | `ret i32 %val` / `ret void` | 返回值 |
| `add` | `add i32 %a, %b` | 加法 |
| `sub` | `sub i32 %a, %b` | 减法 |
| `mul` | `mul i32 %a, %b` | 乘法 |
| `sdiv` | `sdiv i8 %a, %b` | 有符号除法 |
| `udiv` | `udiv i8 %a, %b` | 无符号除法 |
| `alloca` | `%p = alloca i32` | 栈分配 |
| `store` | `store i32 42, ptr %p` | 写入 |
| `load` | `%v = load i32, ptr %p` | 读取 |
| `getelementptr` | `gep %T, ptr %p, i32 0, i32 1` | 地址运算 |
| `br` | `br i1 %c, label %t, label %f` | 分支 |
| `phi` | `phi i32 [%v1, %b1], [%v2, %b2]` | φ 节点 |
| `call` | `call i32 @func(i32 %a)` | 调用 |
| `trunc..to` | `trunc i32 257 to i8` | 截断 |
| `zext..to` | `zext i8 200 to i32` | 零扩展 |
| `sext..to` | `sext i8 -1 to i32` | 符号扩展 |
| `icmp` | `icmp sgt i32 %a, %b` | 整数比较 |
| `select` | `select i1 %c, i32 %a, i32 %b` | 三目运算 |

## icmp 谓词

`eq`, `ne`, `sgt`, `sge`, `slt`, `sle`, `ugt`, `uge`, `ult`, `ule`

## 重要规则

1. **SSA**: 每个局部变量（`%`）只能赋值一次
2. **类型匹配**: 指令操作数类型必须一致
3. **基本块**: 最后一个指令必须是终结指令（`ret`/`br`/`switch`）
4. **φ 位置**: φ 节点必须在基本块开头
5. **结构体 GEP**: 字段索引必须是常量
6. **不透明指针**: LLVM 15+ 使用 `ptr`

## 调试

```bash
llc -verify-machineinstrs file.ll       # 验证 IR
opt -O2 -S file.ll                      # 查看优化后 IR
opt -load-pass-plugin lib.dylib --help-hidden | grep "pass-name"  # 查看可用 Pass
clang -S -emit-llvm -O0 test.c          # C → IR
```
