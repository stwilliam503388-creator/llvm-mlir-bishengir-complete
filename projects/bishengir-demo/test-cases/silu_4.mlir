// ==- silu_4.mlir - SiLU / Swish 激活函数 -==//
// silu(x) = x * sigmoid(x)
// 用途: LLaMA / GPT 等现代 Transformer 模型
// 对应 AscendNPU-IR: 组合 arith.mulf + math.exp + arith.divf
// 降级: 7 行 → 23 行 LLVM

module {
  func.func @silu(%A: memref<4xf32>, %B: memref<4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %neg = arith.negf %a : f32
      %exp = math.exp %neg : f32
      %c1 = arith.constant 1.0 : f32
      %one = arith.addf %exp, %c1 : f32
      %sig = arith.divf %c1, %one : f32
      %silu = arith.mulf %a, %sig : f32
      linalg.yield %silu : f32
    }
    return
  }
}
