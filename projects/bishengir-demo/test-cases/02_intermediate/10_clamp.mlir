// Clamp — 数值裁剪
// 公式: y = clamp(x, min, max) = min(max(x, min), max)
// 一句话: 超过范围的值截断到边界
// 专业角色: 梯度裁剪 (LLM 训练), 量化前处理
// 用在哪: 梯度裁剪 / 量化前处理 / 强化学习
// 降级: 两次 cmpf + select
// bishengir: 分段线性函数映射

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.cmpf ogt
// CHECK: arith.select

module {
  func.func @clamp(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %min = arith.constant -1.0 : f32
    %max = arith.constant 1.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %c1 = arith.cmpf olt, %a, %min : f32
      %clamp_min = arith.select %c1, %min, %a : f32
      %c2 = arith.cmpf ogt, %clamp_min, %max : f32
      %clamped = arith.select %c2, %max, %clamp_min : f32
      linalg.yield %clamped : f32
    }
    return
  }
}
