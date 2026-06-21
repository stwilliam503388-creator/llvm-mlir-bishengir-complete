// ReLU — 线性整流激活函数
//
// 功能: y = max(0, x), 负值截断为0
// AI 角色: 全模型通用的激活函数
//   CNN 全系标配, 计算量最低, GPU 友好. LLM 早期用 ReLU, 后被 GELU 替代.
// 应用场景: ResNet/VGG/CNN 全系
// MLIR 模式: arith.cmpf + arith.select, 条件分支
// 对应 bishengir: hfusion.elemwise_unary {fun = relu}
//
1|// ==- relu_4x4.mlir - ReLU 激活函数 -==//
2|//
3|// y = max(x, 0) — 最常用的激活函数，视觉和语言模型都有
4|// 对应 AscendNPU-IR: bishengir 通过 linalg.generic 做逐元素操作
5|// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
6|// 等价 bishengir-opt: --convert-linalg-to-hfusion
7|// 预期 bishengir 输出: hfusion.elemwise_unary {fun = relu}
8|//
9|// 降级效果:  5 行输入 → affine (约 20 行) → LLVM (约 40 行)
10|
11|module {
12|  func.func @relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
13|    %c0 = arith.constant 0.0 : f32
14|    linalg.generic {
15|      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>],
16|      iterator_types = ["parallel", "parallel"]
17|    } ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
18|    ^bb0(%a: f32, %b: f32):
19|      %gt = arith.cmpf ogt, %a, %c0 : f32
20|      %relu = arith.select %gt, %a, %c0 : f32
21|      linalg.yield %relu : f32
22|    }
23|    return
24|  }
25|}
26|