# 用例 2 — fusion-add-mul

两个 linalg.generic（先 add 后 mul）→ hivm 中用 Unified Buffer 复用中间结果。

## 融合的关键

| 未融合（如果各写回 HBM） | 融合后 |
|------------------------|--------|
| add → store 到 gm | add 结果留在 ub |
| load 从 gm → mul | 直接 mul 用 ub 中的结果 |
| 4 次 HBM↔UB 搬运 | 2 次搬运（省 50%） |

**融合的本质**：hivm 在 Unified Buffer 中直接把 add 的结果传给 mul，不经过 Global Memory。

→ 用例 3：hivm 内部更细的操作
