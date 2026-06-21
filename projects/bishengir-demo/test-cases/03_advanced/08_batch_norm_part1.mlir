// Batch Norm (Part 1) — 批归一化: 均值计算
// 公式: mean[j] = sum_i x[i][j] / N
// 一句话: 算一批数据在各通道的平均水平
// 专业角色: BN 第一步, 稳定训练, 允许高学习率
// 用在哪: ResNet/CNN 全系 (训练阶段)
// 降级: reduction + parallel 混合 iterator
// bishengir: 拆为 reduce + broadcast + elemwise 三步
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.addf

module {
  func.func @bn_mean(%A: memref<4x4xf32>, %mean: memref<4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>], iterator_types = ["reduction", "parallel"]} ins(%A : memref<4x4xf32>) outs(%mean : memref<4xf32>) {
    ^bb0(%a: f32, %m: f32):
      %sum = arith.addf %m, %a : f32
      linalg.yield %sum : f32
    }
    return
  }
}
