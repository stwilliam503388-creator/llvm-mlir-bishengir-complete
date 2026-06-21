// ReLU — 线性整流激活函数
// 公式: y = max(0, x)
// 一句话: 负数变0, 正数不变
// 专业角色: 最常见的激活函数, 引入非线性, 计算成本极低
// 用在哪: ResNet/VGG/CNN 全系列
// 降级: arith.cmpf (比较) + arith.select (选择)
// bishengir: hfusion.elemwise_unary {fun = relu}

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.cmpf ogt
// CHECK: arith.select

module {
  func.func @relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %gt = arith.cmpf ogt, %a, %c0 : f32
      %relu = arith.select %gt, %a, %c0 : f32
      linalg.yield %relu : f32
    }
    return
  }
}
