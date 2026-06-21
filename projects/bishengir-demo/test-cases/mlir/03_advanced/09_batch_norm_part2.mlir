// Batch Norm (Part 2) — 批归一化: 标准化
// 公式: y = gamma * (x - mu) / sqrt(var + eps) + beta
// 一句话: 减去均值, 除以标准差, 缩放平移
// 专业角色: BN/LN/IN/GN 四种归一化的通用第二步
// 用在哪: ResNet/CNN 全系 (归一化通用模式)
// 降级: 5个ins的linalg.generic + math.sqrt
// bishengir: 拆为 elemwise + math 组合
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf %s
// CHECK: affine.for
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: arith.addf

module {
  func.func @bn_norm(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %mean: memref<4xf32>, %var: memref<4xf32>, %gamma: memref<4xf32>, %beta: memref<4xf32>) {
    %eps = arith.constant 1.0e-5 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A, %mean, %var, %gamma, %beta : memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %m: f32, %v: f32, %g: f32, %bt: f32, %b: f32):
      %norm = arith.subf %a, %m : f32
      %std = arith.addf %v, %eps : f32
      %sqrt = math.sqrt %std : f32
      %div = arith.divf %norm, %sqrt : f32
      %scaled = arith.mulf %div, %g : f32
      %out = arith.addf %scaled, %bt : f32
      linalg.yield %out : f32
    }
    return
  }
}
