# 用例 4 — hivm-to-llvm

hivm → LLVM IR。最后一步 Lowering，回到你熟悉的 LLVM 世界。

## 关键变化

| hivm | LLVM |
|------|------|
| `hivm.load` | `llvm.load` |
| `hivm.vadd` | `llvm.fadd` |
| `hivm.store` | `llvm.store` |
| Ascend 专有指令 | 通用 LLVM 指令 |

> 所有 MLIR 的 Dialect，最终都要 Lowering 到 `llvm` dialect。这就是 MLIR 和 LLVM 的关系。

## 学完后

→ 用例 5：完整 Pipeline 追踪
