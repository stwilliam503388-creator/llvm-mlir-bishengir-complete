// Layer Norm — 层归一化 (平方差部分)
//
// 功能: y = (x - mean) / sqrt(var + eps), 归一化计算
// AI 角色: Transformer 标配归一化. 对每个 token 自己做归一化, 不依赖 batch.
//   GPT/BERT/LLaMA 每层 Attention 和 FFN 后都有 LN.
//   与 BN 区别: LN 适合变长序列 (NLP/LLM).
// 应用场景: Transformer 全系 (GPT/BERT/LLaMA)
// MLIR 模式: linalg.generic + subf + mulf, 逐元素
// 对应 bishengir: 组合模式, 需配合 reduce + broadcast
//
1|// ==- layer_norm_4x4.mlir - Layer Normalization -==//
2|// LN: y = (x - mean) / sqrt(var + eps) * gamma + beta
3|// 用途: Transformer 每一层后的归一化
4|// 对应 AscendNPU-IR: 需 reduction + broadcast + elemwise 组合
5|// 本文件演示: 归一化计算的平方差部分
6|// 降级: 7 行 → 44 行 LLVM
7|
8|module {
9|  func.func @layer_norm(%A: memref<4x4xf32>, %mean: memref<f32>, %var: memref<f32>, %B: memref<4x4xf32>) {
10|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
11|    ^bb0(%a: f32, %b: f32):
12|      %diff = arith.subf %a, %b : f32
13|      %sq = arith.mulf %diff, %diff : f32
14|      linalg.yield %sq : f32
15|    }
16|    return
17|  }
18|}
19|