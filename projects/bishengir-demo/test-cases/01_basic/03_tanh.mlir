// Tanh — 双曲正切激活函数
// 公式: y = tanh(x), 输出范围 (-1, 1)
// 一句话: 把任意数压缩到 -1~1 之间
// 专业角色: RNN/LSTM 的门控信号, GELU 近似用到此函数
// 用在哪: RNN/LSTM / GELU 计算中间步骤
// 降级: math.tanh 内建函数
// bishengir: hfusion.elemwise_unary {fun = tanh}

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf %s
// CHECK: affine.for
// CHECK: math.tanh

module {
  func.func @tanh(%A: memref<4xf32>, %B: memref<4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %th = math.tanh %a : f32
      linalg.yield %th : f32
    }
    return
  }
}
