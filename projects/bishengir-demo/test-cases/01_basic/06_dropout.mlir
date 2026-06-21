// Dropout — 随机丢弃 (训练正则化)
// 公式: y = x * scale (简化版)
// 一句话: 训练时随机让部分神经元"翘课"
// 专业角色: 正则化技术, 防止过拟合
// 用在哪: 全连接层 / Transformer 训练阶段
// 降级: arith.mulf 逐元素乘法
// bishengir: hfusion.elemwise_binary {fun = mul}

module {
  func.func @dropout(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %scale: f32) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %scaled = arith.mulf %a, %scale : f32
      linalg.yield %scaled : f32
    }
    return
  }
}
