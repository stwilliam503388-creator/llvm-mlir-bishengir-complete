// Reduce Max — 最大值归约
//
// 功能: max = max(all elements), 求矩阵最大值
// AI 角色: Softmax 数值稳定性关键. 先减最大值防止 exp 溢出 (exp(1000)=inf).
//   也用于 Max Pooling / Attention Top-K 筛选.
// 应用场景: Softmax 数值稳定 / Max Pooling
// MLIR 模式: reduction + arith.cmpf + arith.select
// 对应 bishengir: hfusion.reduce {fun = max}
//
1|// ==- reduce_max_4x4.mlir - 归约求最大值 -==//
2|//
3|// max = max(x[i][j]) — 求矩阵最大值（用于 softmax 数值稳定）
4|// 对应 AscendNPU-IR: hfusion.reduce {fun = max}
5|// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
6|
7|module {
8|  func.func @reduce_max(%A: memref<4x4xf32>, %init: memref<f32>) {
9|    linalg.generic {
10|      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>],
11|      iterator_types = ["reduction", "reduction"]
12|    } ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
13|    ^bb0(%a: f32, %b: f32):
14|      %gt = arith.cmpf ogt, %a, %b : f32
15|      %max = arith.select %gt, %a, %b : f32
16|      linalg.yield %max : f32
17|    }
18|    return
19|  }
20|}
21|