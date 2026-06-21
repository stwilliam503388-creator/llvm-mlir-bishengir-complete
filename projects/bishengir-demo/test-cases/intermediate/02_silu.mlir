// SiLU / Swish — 门控激活函数
//
// 功能: silu(x) = x * sigma(x), 输入×sigmoid(输入)
// AI 角色: LLaMA/Mistral/Gemma 等现代 LLM 的 FFN 层标准激活函数
//   使用 SwiGLU: SiLU(x) × y. 比 GELU 计算量低, 效果相当.
// 应用场景: LLaMA/Mistral/Gemma FFN 层 (2024-2025 最常用激活函数)
// MLIR 模式: sigmoid + mulf, 5步组合
// 对应 bishengir: 组合模式可融合
//
1|// ==- silu_4.mlir - SiLU / Swish 激活函数 -==//
2|// silu(x) = x * sigmoid(x)
3|// 用途: LLaMA / GPT 等现代 Transformer 模型
4|// 对应 AscendNPU-IR: 组合 arith.mulf + math.exp + arith.divf
5|// 降级: 7 行 → 23 行 LLVM
6|
7|module {
8|  func.func @silu(%A: memref<4xf32>, %B: memref<4xf32>) {
9|    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
10|    ^bb0(%a: f32, %b: f32):
11|      %neg = arith.negf %a : f32
12|      %exp = math.exp %neg : f32
13|      %c1 = arith.constant 1.0 : f32
14|      %one = arith.addf %exp, %c1 : f32
15|      %sig = arith.divf %c1, %one : f32
16|      %silu = arith.mulf %a, %sig : f32
17|      linalg.yield %silu : f32
18|    }
19|    return
20|  }
21|}
22|