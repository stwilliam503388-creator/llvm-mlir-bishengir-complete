// Hard Sigmoid — 硬 Sigmoid (线性近似)
//
// 功能: hard_sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)
// AI 角色: 轻量化模型激活函数. 用分段线性近似 sigmoid, 计算量小 3 倍.
//   适合移动端部署 (MobileNetV3), 没有 math.exp 开销.
// 应用场景: MobileNetV3 / 轻量化 CNN
// MLIR 模式: maximumf + minimumf 数值裁剪
// 对应 bishengir: 分段线性函数映射
//
1|// ==- hard_sigmoid_4.mlir - Hard Sigmoid 激活 -==//
2|// hard_sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)
3|// 用途: MobileNet 等轻量化模型（计算量比 sigmoid 小）
4|// 对应 AscendNPU-IR: arith.mulf + addf + max/minimumf
5|// 降级: 7 行 → 39 行 LLVM
6|
7|module {
8|  func.func @hard_sigmoid(%A: memref<4xf32>, %B: memref<4xf32>) {
9|    %c0 = arith.constant 0.0 : f32
10|    %c1 = arith.constant 1.0 : f32
11|    %c05 = arith.constant 0.2 : f32
12|    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A : memref<4xf32>) outs(%B : memref<4xf32>) {
13|    ^bb0(%a: f32, %b: f32):
14|      %mul = arith.mulf %a, %c05 : f32
15|      %add = arith.addf %mul, %c05 : f32
16|      %clamped = arith.maximumf %add, %c0 : f32
17|      %out = arith.minimumf %clamped, %c1 : f32
18|      linalg.yield %out : f32
19|    }
20|    return
21|  }
22|}
23|