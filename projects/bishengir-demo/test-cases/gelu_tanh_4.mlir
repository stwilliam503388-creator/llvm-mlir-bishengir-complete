// ==- gelu_tanh_4.mlir - GELU 近似激活（tanh 版本）-==//
// gelu(x) ≈ 0.5 * x * (1 + tanh(x))
// 用途: BERT/GPT 等 Transformer 模型
// 对应 AscendNPU-IR: math.tanh + arith.mulf/addf
// 降级: 6 行 → 38 行 LLVM

module {
  func.func @gelu_tanh(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c1 = arith.constant 1.0 : f32
    %c05 = arith.constant 0.5 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %th = math.tanh %a : f32
      %plus1 = arith.addf %th, %c1 : f32
      %gelu = arith.mulf %c05, %plus1 : f32
      %result = arith.mulf %a, %gelu : f32
      linalg.yield %result : f32
    }
    return
  }
}
