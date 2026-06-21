// GELU (tanh 近似) — 高斯误差线性单元
// 公式: gelu(x) ~= 0.5 * x * (1 + tanh(x))
// 一句话: 平滑版 ReLU, 负数区逐渐变0
// 专业角色: BERT/GPT-2/GPT-3 FFN 层标准激活函数
// 用在哪: BERT/RoBERTa/GPT-2/GPT-3 FFN 层
// 降级: math.tanh + addf + mulf, 4步
// bishengir: 组合后可融合

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
