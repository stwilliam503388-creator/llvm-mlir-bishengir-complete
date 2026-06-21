// Tanh — 双曲正切激活函数
//
// 功能: y = tanh(x), S 形函数, 输出范围 (-1, 1)
// AI 角色: RNN/LSTM 经典门控激活函数, 用于控制信息流
// 应用场景: RNN/LSTM, GELU 计算中间步骤
// MLIR 模式: math.tanh 内建函数调用
// 对应 bishengir: hfusion.elemwise_unary {fun = tanh}
//
1|// ==- tanh_4.mlir - Tanh 激活函数 -==//
2|//
3|// y = tanh(x) — RNN/Transformer 中常用激活
4|// 对应 AscendNPU-IR: 逐元素 math.tanh
5|// 等价 bishengir-opt: --convert-linalg-to-hfusion
6|
7|module {
8|  func.func @tanh(%A: memref<4xf32>, %B: memref<4xf32>) {
9|    linalg.generic {
10|      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
11|      iterator_types = ["parallel"]
12|    } ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
13|    ^bb0(%a: f32, %b: f32):
14|      %th = math.tanh %a : f32
15|      linalg.yield %th : f32
16|    }
17|    return
18|  }
19|}
20|