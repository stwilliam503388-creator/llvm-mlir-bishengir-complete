// Sigmoid — Logistic 激活函数
// 公式: sigma(x) = 1 / (1 + e^{-x}), 输出 (0, 1)
// 一句话: 把任意数压缩到 0~1 之间
// 专业角色: 二分类输出层, RNN 门控, SwiGLU (LLaMA) 中间步骤
// 用在哪: 二分类 / RNN门控 / SwiGLU
// 降级: negf + exp + addf + divf, 4步
// bishengir: 组合后可融合

module {
  func.func @sigmoid(%A: memref<4xf32>, %B: memref<4xf32>) {
    %c1 = arith.constant 1.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %neg = arith.negf %a : f32
      %exp = math.exp %neg : f32
      %one = arith.addf %exp, %c1 : f32
      %sig = arith.divf %c1, %one : f32
      linalg.yield %sig : f32
    }
    return
  }
}
