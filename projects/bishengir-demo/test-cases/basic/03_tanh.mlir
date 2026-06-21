// ==- tanh_4.mlir - Tanh 激活函数 -==//
//
// y = tanh(x) — RNN/Transformer 中常用激活
// 对应 AscendNPU-IR: 逐元素 math.tanh
// 等价 bishengir-opt: --convert-linalg-to-hfusion

module {
  func.func @tanh(%A: memref<4xf32>, %B: memref<4xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %th = math.tanh %a : f32
      linalg.yield %th : f32
    }
    return
  }
}
