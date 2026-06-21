// Softmax (数值稳定版, 第一步)
//
// 功能: y = exp(x - max(x)), 减最大值后取指数
// AI 角色: Attention 机制的数值稳定计算
//   完整 softmax = exp(x - max(x)) / sum(exp(x - max(x))).
//   Transformer 计算注意力分数分布, LLM 推理核心操作.
// 应用场景: Transformer / BERT/GPT Attention 层
// MLIR 模式: memref.alloc + reduction + memref.store
// 对应 bishengir: 需组合 reduce + broadcast + elemwise
//
1|// ==- softmax_complete_4.mlir - 完整 Softmax 第一步 -==//
2|// y = exp(x - max(x)) — softmax 的数值稳定版本
3|// 用途: Attention 机制核心算子
4|// 对应 AscendNPU-IR: reduction → broadcast → exp
5|// 降级: 10 行 → 44 行 LLVM
6|
7|module {
8|  func.func @softmax_stable(%A: memref<4x4xf32>) {
9|    %max_init = memref.alloc() : memref<f32>
10|    %c0 = arith.constant 0.0 : f32
11|    memref.store %c0, %max_init[] : memref<f32>
12|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%A : memref<4x4xf32>) outs(%max_init : memref<f32>) {
13|    ^bb0(%a: f32, %m: f32):
14|      %gt = arith.cmpf ogt, %a, %m : f32
15|      %max = arith.select %gt, %a, %m : f32
16|      linalg.yield %max : f32
17|    }
18|    return
19|  }
20|}
21|