// Leaky ReLU — 带泄漏的线性整流
//
// 功能: y = x if x > 0 else 0.01*x, 负数侧有微小斜率
// AI 角色: 解决 ReLU 死亡问题 (负数区梯度不为零), GAN 标配激活函数
// 应用场景: GAN / 部分传统 CNN
// MLIR 模式: cmpf + mulf + arith.select
// 对应 bishengir: 条件分支模式映射
//
1|// ==- leaky_relu_4.mlir - Leaky ReLU 激活 -==//
2|// leaky_relu(x) = x if x > 0 else 0.01*x
3|// 用途: 解决 ReLU 死亡问题，GAN 等模型常用
4|// 对应 AscendNPU-IR: arith.cmpf + select 模式
5|// 降级: 7 行 → 37 行 LLVM
6|
7|module {
8|  func.func @leaky_relu(%A: memref<4xf32>, %B: memref<4xf32>) {
9|    %c0 = arith.constant 0.0 : f32
10|    %alpha = arith.constant 0.01 : f32
11|    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
12|    ^bb0(%a: f32, %b: f32):
13|      %gt = arith.cmpf ogt, %a, %c0 : f32
14|      %leak = arith.mulf %a, %alpha : f32
15|      %relu = arith.select %gt, %a, %leak : f32
16|      linalg.yield %relu : f32
17|    }
18|    return
19|  }
20|}
21|