# 用例 3 — hivm 内部结构

hivm 的每个操作都有固定模式：alloc → load → compute → store。

## 三步模式

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1. load | `hivm.hir.load gm → ub` | 从 HBM 搬到片上 |
| 2. compute | `hivm.hir.vadd` | 在 Vector Unit 上计算 |
| 3. store | `hivm.hir.store ub → gm` | 结果写回 HBM |

这是所有 Ascend kernel 的通用模板。

→ 用例 4：hivm.hir → llvm
