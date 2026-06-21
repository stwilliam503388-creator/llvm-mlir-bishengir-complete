// PReLU — 参数化线性整流
//
// 功能: y = x if x > 0 else alpha*x, alpha 可训练 (梯度下降学习)
// AI 角色: 可学习参数的激活函数. 超分辨率模型 (SRGAN) 常用.
// 应用场景: 图像超分辨率 / 精细图像任务
// MLIR 模式: arith.mulf + cmpf + select
// 对应 bishengir: 条件分支模式映射
//
1|// ==- prelu_4x4.mlir - PReLU 激活 -==//
2|// prelu(x) = x if x > 0 else alpha*x (alpha 可学习)
3|// 用途: 图像分类/超分辨率
4|// 降级: 7 行 → 52 行 LLVM
5|
6|module {
7|  func.func @prelu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
8|    %c0 = arith.constant 0.0 : f32
9|    %alpha = arith.constant 0.25 : f32
10|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<4x4xf32>) outs(%B : memref<4x4xf32>) {
11|    ^bb0(%a: f32, %b: f32):
12|      %neg = arith.mulf %a, %alpha : f32
13|      %gt = arith.cmpf ogt, %a, %c0 : f32
14|      %p = arith.select %gt, %a, %neg : f32
15|      linalg.yield %p : f32
16|    }
17|    return
18|  }
19|}
20|