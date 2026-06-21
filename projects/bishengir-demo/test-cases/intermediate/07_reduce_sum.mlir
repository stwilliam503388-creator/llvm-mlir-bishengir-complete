// Reduce Sum — 求和归约
//
// 功能: sum = Sigma_i Sigma_j x[i][j], 矩阵所有元素求和
// AI 角色: 归约操作基础. Layer Norm 分母 / Softmax 分母 / Attention 聚合.
//   reduction = 最常见的 AI 算子之一.
// 应用场景: Layer Norm / Softmax / 聚合操作
// MLIR 模式: linalg.generic + reduction iterator, 多维→标量
// 对应 bishengir: hfusion.reduce {fun = add}
//
1|// ==- reduce_sum_4x4.mlir - 归约求和 -==//
2|//
3|// sum = Σx[i][j] — 把 4x4 矩阵归约为标量
4|// 对应 AscendNPU-IR:
5|//   hfusion.reduce {fun = add, axes = [0, 1]}
6|// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
7|// 等价 bishengir-opt: --convert-linalg-to-hfusion
8|//
9|// MLIR 关键概念: iterator_types = ["reduction", "reduction"]
10|//   "parallel" = 输出索引与输入索引一致
11|//   "reduction" = 输出索引比输入少一维（归约）
12|
13|module {
14|  func.func @reduce_sum(%A: memref<4x4xf32>, %init: memref<f32>) {
15|    linalg.generic {
16|      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>],
17|      iterator_types = ["reduction", "reduction"]
18|    } ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
19|    ^bb0(%a: f32, %b: f32):
20|      %sum = arith.addf %a, %b : f32
21|      linalg.yield %sum : f32
22|    }
23|    return
24|  }
25|}
26|