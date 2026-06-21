// Softmax (exp) — Softmax 指数部分
// 公式: y = e^{x}, 指数运算
// 一句话: 把分数换算成正数, 放大差距
// 专业角色: Attention 机制核心, softmax(QK^T / sqrt(d))
// 用在哪: Transformer Attention / 多分类输出层
// 降级: math.exp 内建函数
// bishengir: 逐元素 math 操作映射

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf %s
// CHECK: affine.for
// CHECK: math.exp

module {
  func.func @softmax_exp(%A: memref<4xf32>, %B: memref<4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %exp = math.exp %a : f32
      linalg.yield %exp : f32
    }
    return
  }
}
