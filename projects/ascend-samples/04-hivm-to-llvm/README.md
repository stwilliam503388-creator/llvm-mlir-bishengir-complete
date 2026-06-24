# 用例 4 — hivm→llvm

最后一步 Lowering：hivm.hir Ascend 专用指令 → 通用 LLVM IR。

## 映射关系

| hivm.hir | LLVM |
|----------|------|
| `memref.alloc` | `llvm.alloca` |
| `hivm.hir.load gm→ub` | `llvm.load + llvm.store` |
| `hivm.hir.vadd` | `llvm.fadd` |
| `hivm.hir.store ub→gm` | `llvm.store` |
| `hacc.entry` | （丢弃） |

> 所有 MLIR Dialect 最终都要 Lowering 到 `llvm` dialect。hivm.hir 也不例外。

→ 用例 5：完整 Pipeline
