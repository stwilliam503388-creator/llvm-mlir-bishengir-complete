// Hard Sigmoid — 硬 Sigmoid (分段线性近似)
// 公式: hard_sigmoid(x) = clamp(0.2x + 0.5, 0, 1)
// 一句话: 用折线近似 S 形, 省去 exp 计算
// 专业角色: 轻量化模型激活函数, 计算量小 3 倍
// 用在哪: MobileNetV3 / 轻量化 CNN
// 降级: maximumf + minimumf 数值裁剪
// bishengir: 分段线性函数映射

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: arith.maximumf
// CHECK: arith.minimumf

module {
  func.func @hard_sigmoid(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %c05 = arith.constant 0.2 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %mul = arith.mulf %a, %c05 : f32
      %add = arith.addf %mul, %c05 : f32
      %clamped = arith.maximumf %add, %c0 : f32
      %out = arith.minimumf %clamped, %c1 : f32
      linalg.yield %out : f32
    }
    return
  }
}
