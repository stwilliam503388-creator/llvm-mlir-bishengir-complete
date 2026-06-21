// Batch Norm (Part 2) — 批归一化标准化
//
// 功能: y = gamma * (x - mu) / sqrt(var + eps) + beta
// AI 角色: BN 第二步 — 减去均值, 除以标准差, 缩放平移.
//   这个 design pattern 在所有归一化层 (BN/LN/IN/GN) 中通用.
// 应用场景: ResNet/CNN 全系 + 归一化层通用模式
// MLIR 模式: 5个ins操作数的 linalg.generic + math.sqrt
// 对应 bishengir: 需拆解为 elemwise + math 组合
//
1|// ==- batch_norm_4x4_part2.mlir - Batch Norm (normalize) -==//
2|// BN 第二步: (x - mean) / sqrt(var + eps) * gamma + beta
3|// Broadcasting mean/var/gamma/beta 到每个元素
4|// 降级: 9 行 → 87 行 LLVM
5|
6|module {
7|  func.func @bn_norm(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %mean: memref<4xf32>, %var: memref<4xf32>, %gamma: memref<4xf32>, %beta: memref<4xf32>) {
8|    %eps = arith.constant 1.0e-5 : f32
9|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A, %mean, %var, %gamma, %beta : memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%B : memref<4x4xf32>) {
10|    ^bb0(%a: f32, %m: f32, %v: f32, %g: f32, %bt: f32, %b: f32):
11|      %norm = arith.subf %a, %m : f32
12|      %std = arith.addf %v, %eps : f32
13|      %sqrt = math.sqrt %std : f32
14|      %div = arith.divf %norm, %sqrt : f32
15|      %scaled = arith.mulf %div, %g : f32
16|      %out = arith.addf %scaled, %bt : f32
17|      linalg.yield %out : f32
18|    }
19|    return
20|  }
21|}
22|