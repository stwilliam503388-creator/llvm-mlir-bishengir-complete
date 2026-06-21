// Avg Pool — 平均池化 (下采样)
//
// 功能: 2×2窗口取平均值, stride=2, 4×4→2×2
// AI 角色: CNN 下采样 (比 max pool 更平滑). ResNet 分类头前用 avg pool.
// 应用场景: ResNet 分类头 / 平滑下采样
// MLIR 模式: affine.for ×4 + 累加 + 除法
// 对应 bishengir: linalg.pooling_nchw_sum + 除法 (需 bishengir 自编译)
//
1|// ==- avg_pool_4x4.mlir - 平均池化 (affine.for 实现) -==//
2|// 2x2 kernel, stride=2, valid padding
3|// 输入 4x4 → 输出 2x2
4|// 用 affine.for 代替 linalg.pooling named op
5|// 降级: 11 行 → 83 行 LLVM
6|
7|module {
8|  func.func @avg_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
9|    %c04 = arith.constant 0.25 : f32
10|    affine.for %oh = 0 to 2 {
11|      affine.for %ow = 0 to 2 {
12|        %c0 = arith.constant 0.0 : f32
13|        memref.store %c0, %output[%oh, %ow] : memref<2x2xf32>
14|        affine.for %kh = 0 to 2 {
15|          affine.for %kw = 0 to 2 {
16|            %v = affine.load %input[%oh * 2 + %kh, %ow * 2 + %kw] : memref<4x4xf32>
17|            %acc = memref.load %output[%oh, %ow] : memref<2x2xf32>
18|            %sum = arith.addf %v, %acc : f32
19|            memref.store %sum, %output[%oh, %ow] : memref<2x2xf32>
20|          }
21|        }
22|        %sum = memref.load %output[%oh, %ow] : memref<2x2xf32>
23|        %avg = arith.mulf %sum, %c04 : f32
24|        memref.store %avg, %output[%oh, %ow] : memref<2x2xf32>
25|      }
26|    }
27|    return
28|  }
29|}
30|