# 用例 2 — fusion-add-mul

两个独立 linalg.generic（add + mul）融合为一个 husion.elemwise_binary "add_mul"。

30 行 → 5 行。10 倍压缩。这就是融合的力量。

## 关键点

| input.mlir | expected.mlir |
|-----------|--------------|
| 2 个 linalg.generic | 1 个 husion.elemwise_binary |
| add 和 mul 各写一次中间结果 | 一次读写完成 |
| 4 次数据搬运（HBM↔L1） | 2 次数据搬运 |

## 学完后

→ 用例 3：husion → hivm 的 Lowering
