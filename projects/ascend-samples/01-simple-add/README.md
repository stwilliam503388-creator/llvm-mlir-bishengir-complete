# 用例 1 — simple-add

最简逐元素加法。linalg.generic 14 行 → husion.elemwise_binary 3 行。

## 逐行解读

| input.mlir (linalg) | 含义 |
|---------------------|------|
| `linalg.generic` | 泛型线性代数操作 |
| `affine_map<(d0) -> (d0)>` | 逐元素映射（每个位置独立） |
| `iterator_types = ["parallel"]` | 所有维度可并行 |
| `arith.addf` | 核心操作：浮点加法 |

| expected.mlir (husion) | 含义 |
|------------------------|------|
| `husion.elemwise_binary "add"` | 逐元素二元操作，类型为 add |

## 学完后

→ 用例 2：两个操作融合
