---
tags:
  - 工程
---

# LLVM 类型系统与 GEP

## 类型系统

| 类型 | 示例 | 说明 |
|------|------|------|
| `void` | `ret void` | 无返回值 |
| `iN` | `i1`, `i8`, `i32`, `i64` | 任意位宽整型 |
| `float/double` | `fadd float %a, %b` | IEEE 754 |
| `ptr` | `store i32 42, ptr %p` | 不透明指针（LLVM 15+） |
| `[N x T]` | `[10 x i32]` | 定长数组 |
| `{T1, T2}` | `{i32, ptr}` | 结构体 |
| `<N x T>` | `<4 x float>` | 向量（SIMD） |

## 有符号/无符号：在指令，不在类型

```llvm
%1 = udiv i8 -6, 2    ; 无符号: (256-6)/2 = 125
%2 = sdiv i8 -6, 2    ; 有符号: (-6)/2 = -3
```

## 类型转换

```llvm
%t = trunc i32 257 to i8      ; 截断: 257 → 1
%z = zext i8 200 to i32       ; 零扩展（高位补0）
%s = sext i8 -1 to i32        ; 符号扩展（高位补符号位）
```

## GEP (GetElementPtr) — LLVM 最困惑的指令

**GEP 不做内存访问，只计算地址。**

语法：`gep <元素类型>, ptr <基址>, <索引1>, <索引2>, ...`

### 结构体 GEP

```llvm
%Point = type { i32, i32 }
%p = alloca %Point
%x_ptr = getelementptr %Point, ptr %p, i32 0, i32 0  ; p.x
%y_ptr = getelementptr %Point, ptr %p, i32 0, i32 1  ; p.y
```

- 索引 0：解引用指针本身
- 结构体字段索引：**必须是编译期常量**

### 数组 GEP（"剥洋葱"）

```llvm
%matrix = alloca [2 x [3 x i32]]
%elem = getelementptr [2 x [3 x i32]], ptr %matrix, i32 0, i32 1, i32 2
```

| 索引 | 操作 | 说明 |
|------|------|------|
| 0 | `[2x[3xi32]]` → `[3xi32]` | 解引用数组指针 |
| 1 | `[3xi32]` → `i32` | 取第二行 |
| 2 | `i32` | 取第三列 |

### 记忆口诀

> **"剥一层、出一个类型，最后一个索引是你要的元素的偏移量。"**

### GEP 两条铁律

1. 结构体字段索引必须是常量
2. GEP 不访问内存——只是指针运算
