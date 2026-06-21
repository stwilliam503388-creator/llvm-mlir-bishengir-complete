// Clamp — 数值裁剪
//
// 功能: y = clamp(x, min, max), 限制值在 [min, max] 区间
// AI 角色: 梯度裁剪 (LLM 训练防爆炸) / 量化前处理 (INT8 裁剪) / 强化学习 PPO
// 应用场景: 梯度裁剪 / 量化前处理 / 强化学习
// MLIR 模式: 两次 arith.cmpf + arith.select
// 对应 bishengir: 分段线性函数映射
//
1|// ==- clamp_4x4.mlir - 数值裁剪 -==//
2|// clamp(x, min, max) — 将 tensor 值限制在 [min, max] 内
3|// 用途: 梯度裁剪 / 激活值截断
4|// 降级: 7 行 → 53 行 LLVM
5|
6|module {
7|  func.func @clamp(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
8|    %min = arith.constant -1.0 : f32
9|    %max = arith.constant 1.0 : f32
10|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
11|    ^bb0(%a: f32, %b: f32):
12|      %c1 = arith.cmpf olt, %a, %min : f32
13|      %clamp_min = arith.select %c1, %min, %a : f32
14|      %c2 = arith.cmpf ogt, %clamp_min, %max : f32
15|      %clamped = arith.select %c2, %max, %clamp_min : f32
16|      linalg.yield %clamped : f32
17|    }
18|    return
19|  }
20|}
21|