// GEMM + ReLU — 矩阵乘+激活融合
// 公式: y = ReLU(x @ W)
// 一句话: 矩阵乘完立刻激活, 两步并一步
// 专业角色: MLP 层标准融合模式, 减少中间buffer读写
// 用在哪: MLP 层 / FFN 层
// 降级: linalg.matmul + linalg.generic 两阶段
// bishengir: 可融合为单个kernel
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: affine.for
// CHECK: arith.cmpf ogt
// CHECK: arith.select

module {
  func.func @gemm_relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)
    %c0 = arith.constant 0.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%C : memref<4x4xf32>) outs(%C : memref<4x4xf32>) {
    ^bb0(%c: f32, %out: f32):
      %gt = arith.cmpf ogt, %c, %c0 : f32
      %relu = arith.select %gt, %c, %c0 : f32
      linalg.yield %relu : f32
    }
    return
  }
}
