// Max Pool — 最大池化 (下采样)
//
// 功能: 2×2窗口取最大值, stride=2, 4×4→2×2
// AI 角色: CNN 下采样层 — 保留最强激活值, 丢弃位置信息.
//   经典 CNN 标配 (LeNet/AlexNet/VGG). 现代模型用 stride=2 卷积替代.
//   用 affine.for 实现 (linalg.generic 不支持 stride>1).
// 应用场景: LeNet/AlexNet/VGG 下采样层
// MLIR 模式: affine.for ×4 + cmpf + select, 手动循环
// 对应 bishengir: linalg.pooling_nhwc_max (需 bishengir 自编译)
//
1|// ==- max_pool_4x4.mlir - 最大池化 (affine.for 实现) -==//
2|// 2x2 kernel, stride=2, valid padding
3|// 输入 4x4 → 输出 2x2
4|// 用 affine.for 代替 linalg.pooling named op
5|// 原因: Homebrew LLVM 22 未编译 pooling named op
6|// 对应 AscendNPU-IR: bishengir 的 linalg.pooling_nhwc_max
7|// 降级: 9 行 → 80 行 LLVM
8|
9|module {
10|  func.func @max_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
11|    %c0 = arith.constant 0.0 : f32
12|    affine.for %oh = 0 to 2 {
13|      affine.for %ow = 0 to 2 {
14|        %init = memref.load %output[%oh, %ow] : memref<2x2xf32>
15|        affine.for %kh = 0 to 2 {
16|          affine.for %kw = 0 to 2 {
17|            %v = affine.load %input[%oh * 2 + %kh, %ow * 2 + %kw] : memref<4x4xf32>
18|            %gt = arith.cmpf ogt, %v, %init : f32
19|            %max = arith.select %gt, %v, %init : f32
20|            memref.store %max, %output[%oh, %ow] : memref<2x2xf32>
21|          }
22|        }
23|      }
24|    }
25|    return
26|  }
27|}
28|