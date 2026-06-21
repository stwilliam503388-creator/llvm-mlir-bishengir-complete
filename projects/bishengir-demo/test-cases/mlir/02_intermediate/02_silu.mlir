// SiLU / Swish — 门控激活函数
// 公式: silu(x) = x * sigmoid(x)
// 一句话: 输入乘自己的门控值, LLaMA 系列的首选
// 专业角色: SwiGLU = SiLU(x) * y, LLaMA/Mistral/Gemma FFN 层标配
// 用在哪: LLaMA/Mistral/Gemma FFN 层
// 降级: sigmoid + mulf, 5步
// bishengir: 组合后可融合

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf %s
// CHECK: affine.for
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: arith.mulf

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
