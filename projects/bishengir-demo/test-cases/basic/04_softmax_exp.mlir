// Softmax (exp) — Softmax 指数部分
//
// 功能: y = exp(x), 指数运算 (softmax 前半部分)
// AI 角色: Attention 机制核心操作, Transformer 计算注意力分数
// 应用场景: Transformer / BERT/GPT Attention 层
// MLIR 模式: math.exp 内建函数调用
// 对应 bishengir: 逐元素 math 操作映射
//
1|// ==- softmax_4.mlir - Softmax 指数部分 -==//
2|//
3|// y[i] = exp(x[i]) — softmax 的第一步（元素级指数）
4|// 完整 softmax: exp(x - max(x)) / sum(exp(x - max(x)))
5|// 本文件只演示 exp 部分，reduce + div 由其他用例覆盖
6|// 对应 AscendNPU-IR: 逐元素 math 操作映射到 hfusion
7|// 等价 bishengir-opt: --convert-linalg-to-hfusion
8|
9|module {
10|  func.func @softmax_exp(%A: memref<4xf32>, %B: memref<4xf32>) {
11|    linalg.generic {
12|      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
13|      iterator_types = ["parallel"]
14|    } ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
15|    ^bb0(%a: f32, %b: f32):
16|      %exp = math.exp %a : f32
17|      linalg.yield %exp : f32
18|    }
19|    return
20|  }
21|}
22|