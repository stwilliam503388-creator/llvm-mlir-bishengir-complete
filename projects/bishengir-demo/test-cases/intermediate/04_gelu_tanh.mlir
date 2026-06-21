// GELU (tanh 近似) — 高斯误差线性单元
//
// 功能: gelu(x) ≈ 0.5 * x * (1 + tanh(x)), GELU 的 tanh 近似版
// AI 角色: BERT/GPT-2/GPT-3 FFN 层的标准激活函数. 比 ReLU 平滑, 梯度更稳定.
// 应用场景: BERT/RoBERTa/GPT-2/GPT-3 FFN 层
// MLIR 模式: math.tanh + arith.addf + arith.mulf, 4步
// 对应 bishengir: 组合模式可融合
//
1|// ==- gelu_tanh_4.mlir - GELU 近似激活（tanh 版本）-==//
2|// gelu(x) ≈ 0.5 * x * (1 + tanh(x))
3|// 用途: BERT/GPT 等 Transformer 模型
4|// 对应 AscendNPU-IR: math.tanh + arith.mulf/addf
5|// 降级: 6 行 → 38 行 LLVM
6|
7|module {
8|  func.func @gelu_tanh(%A: memref<4xf32>, %B: memref<4xf32>) {
9|    %c1 = arith.constant 1.0 : f32
10|    %c05 = arith.constant 0.5 : f32
11|    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
12|    ^bb0(%a: f32, %b: f32):
13|      %th = math.tanh %a : f32
14|      %plus1 = arith.addf %th, %c1 : f32
15|      %gelu = arith.mulf %c05, %plus1 : f32
16|      %result = arith.mulf %a, %gelu : f32
17|      linalg.yield %result : f32
18|    }
19|    return
20|  }
21|}
22|