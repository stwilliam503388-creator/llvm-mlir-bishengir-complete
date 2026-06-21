# LLVM 内置函数与函数属性

## 内置函数 (Intrinsic Functions)

LLVM 后端保证提供实现，无需每个前端自己实现：

```llvm
; 内存拷贝（对应 memcpy）
call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 16, i1 false)

; 静态分支预测（对应 likely/unlikely）
%pred = call i1 @llvm.expect.i1(i1 %cond, i1 true)

; 陷阱指令
call void @llvm.trap()
```

| 内置函数 | 用途 |
|----------|------|
| `llvm.memcpy` | 内存拷贝 |
| `llvm.memmove` | 内存移动（允许重叠）|
| `llvm.memset` | 内存填充 |
| `llvm.expect` | 静态分支预测 |
| `llvm.trap` | 触发终止 |
| `llvm.debug.declare` | 调试信息 |

## 函数属性

```llvm
attributes #0 = { noinline nounwind optnone ssp uwtable }
```

| 属性 | 含义 |
|------|------|
| `noinline` | 不内联 |
| `nounwind` | 不抛异常 |
| `optnone` | 不做优化 |
| `sanitize_address` | ASan 插桩 |
| `uwtable` | 生成 unwind 表 |
| `readonly` | 不修改内存 |

## 元数据 (Metadata)

以 `!` 开头的行：调试信息、PGO、CFI。

```llvm
!0 = !DILocation(line: 5, column: 3, scope: !1)
```

不需要死记——Clang 自动生成。
