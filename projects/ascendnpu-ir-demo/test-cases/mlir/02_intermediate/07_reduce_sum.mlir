// Reduce Sum — 求和归约
// 公式: sum = sum_i sum_j x[i][j], 多维到标量
// 一句话: 把一堆数加成一个数
// 专业角色: Layer Norm 分母, Softmax 分母, 聚合操作
// 用在哪: Layer Norm / Softmax / 各种聚合
// 降级: linalg.generic + reduction iterator
// bishengir: hfusion.reduce {fun = add}

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.addf

module {
  func.func @reduce_sum(%A: memref<4x4xf32>, %init: memref<f32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%A : memref<4x4xf32>) outs(%init : memref<f32>) {
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }
    return
  }
}
