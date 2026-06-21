// Dropout — 随机丢弃 (训练模式简化版)
//
// 功能: y = x * scale, 按概率缩放 (简化版)
// AI 角色: 防止过拟合的正则化技术. BERT 使用, 现代 LLM 趋向不用.
// 应用场景: 全连接层 / Transformer 训练
// MLIR 模式: arith.mulf 逐元素乘法
// 对应 bishengir: hfusion.elemwise_binary {fun = mul}
//
1|// ==- dropout_4x4.mlir - Dropout（训练模式）-==//
2|// y = x * scale — 简化版，实际还包含 mask
3|// 用途: 防止过拟合，训练时随机丢弃神经元
4|// 降级: 4 行 → 24 行 LLVM
5|
6|module {
7|  func.func @dropout(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %scale: f32) {
8|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
9|    ^bb0(%a: f32, %b: f32):
10|      %scaled = arith.mulf %a, %scale : f32
11|      linalg.yield %scaled : f32
12|    }
13|    return
14|  }
15|}
16|