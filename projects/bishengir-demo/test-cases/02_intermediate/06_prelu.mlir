// PReLU — 参数化线性整流
// 公式: y = x if x > 0 else alpha*x (alpha 可训练)
// 一句话: 让 AI 自己学"负数区斜率应该是多少"
// 专业角色: 超分辨率模型 (SRGAN) 常用
// 用在哪: 图像超分辨率 / 精细图像任务
// 降级: mulf + cmpf + select
// bishengir: 条件分支映射

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: arith.cmpf ogt
// CHECK: arith.select

module {
  func.func @prelu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    %alpha = arith.constant 0.25 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %neg = arith.mulf %a, %alpha : f32
      %gt = arith.cmpf ogt, %a, %c0 : f32
      %p = arith.select %gt, %a, %neg : f32
      linalg.yield %p : f32
    }
    return
  }
}
