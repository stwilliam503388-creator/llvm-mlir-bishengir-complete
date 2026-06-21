// Broadcast — 标量广播到矩阵
//
// 功能: B[i][j] = A, 把标量复制到矩阵每个位置
// AI 角色: 神经网络基础操作 — bias 加法 / 归一化参数广播 / 位置编码
// 应用场景: 所有带 bias/归一化的层
// MLIR 模式: affine_map<(i,j) -> ()>, 标量→矩阵
// 对应 bishengir: hfusion.broadcast
//
1|// ==- broadcast_4x4.mlir - 广播标量到矩阵 -==//
2|//
3|// B[i][j] = A — 把标量 A 广播到 4x4 矩阵
4|// 对应 AscendNPU-IR: hfusion.broadcast
5|// 常见的实际场景: bias 广播 (add bias to matmul output)
6|// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
7|//
8|// MLIR 关键概念: affine_map<(i,j) -> ()>
9|//   "()" 表示输入是标量，与输出 (i,j) 的每个元素都相同
10|//   这比手动写循环更清晰地表达了广播语义
11|
12|module {
13|  func.func @broadcast(%A: memref<f32>, %B: memref<4x4xf32>) {
14|    linalg.generic {
15|      indexing_maps = [affine_map<(i,j) -> ()>, affine_map<(i,j) -> (i,j)>],
16|      iterator_types = ["parallel", "parallel"]
17|    } ins(%A : memref<f32>) outs(%B : memref<4x4xf32>) {
18|    ^bb0(%a: f32, %b: f32):
19|      linalg.yield %a : f32
20|    }
21|    return
22|  }
23|}
24|