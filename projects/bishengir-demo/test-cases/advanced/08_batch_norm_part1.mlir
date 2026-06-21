// Batch Norm (Part 1) — 批归一化均值计算
//
// 功能: mean[j] = sum_i x[i][j] / N, 每个通道求均值
// AI 角色: BN 第一步 — 稳定训练, 允许高学习率, 减少对初始化的依赖.
//   ResNet 每层卷积后都用 BN (Conv→BN→ReLU). LLM 中用 LN 替代 BN.
// 应用场景: ResNet/CNN 全系 (训练阶段)
// MLIR 模式: reduction + parallel 混合 iterator
// 对应 bishengir: 需拆解为 reduce + broadcast + elemwise 组合
//
1|// ==- batch_norm_4x4_part1.mlir - Batch Norm (reduce mean) -==//
2|// BN 第一步: 计算每个通道的均值
3|// 将 4x4 矩阵按列 (i) 归约为 4 个均值
4|// 降级: 5 行 → 48 行 LLVM
5|
6|module {
7|  func.func @bn_mean(%A: memref<4x4xf32>, %mean: memref<4xf32>) {
8|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>], iterator_types = ["reduction", "parallel"]} ins(%A : memref<4x4xf32>) outs(%mean : memref<4xf32>) {
9|    ^bb0(%a: f32, %m: f32):
10|      %sum = arith.addf %m, %a : f32
11|      linalg.yield %sum : f32
12|    }
13|    return
14|  }
15|}
16|