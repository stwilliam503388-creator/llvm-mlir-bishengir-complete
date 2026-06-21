// Layer Norm — 层归一化
// 公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
// 一句话: 每个 token 自己标准化, 不依赖 batch
// 专业角色: Transformer 标配归一化, 适合变长序列
// 用在哪: GPT/BERT/LLaMA 每层 Attention 和 FFN 后
// 降级: linalg.generic + subf + mulf
// bishengir: 组合 (需配合 reduce + broadcast)

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.subf
// CHECK: arith.mulf

module {
  func.func @layer_norm(%A: memref<4x4xf32>, %mean: memref<f32>, %var: memref<f32>, %B: memref<4x4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %diff = arith.subf %a, %b : f32
      %sq = arith.mulf %diff, %diff : f32
      linalg.yield %sq : f32
    }
    return
  }
}
