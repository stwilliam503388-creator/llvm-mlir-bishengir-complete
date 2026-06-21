// Depthwise Conv — 深度可分离卷积
//
// 功能: 逐通道 3×3 卷积, 输入 1×4×4×1, 输出 1×4×4×1
// AI 角色: 深度可分离卷积 = depthwise + pointwise. 计算量约为标准卷积的 1/9~1/3.
//   MobileNet/EfficientNet 核心算子, 适合移动端部署.
// 应用场景: MobileNet/EfficientNet
// MLIR 模式: linalg.depthwise_conv_2d_nhwc_hwcm named op
// 对应 bishengir: named op 映射
//
1|// ==- depthwise_conv_4x4.mlir - 深度可分离卷积 -==//
2|//
3|// 逐通道卷积（depthwise convolution），Mobilenet 的核心算子
4|// 对应 AscendNPU-IR: linalg.depthwise_conv_2d_nhwc_hwcm
5|// 等价 bishengir-opt: --convert-linalg-to-hfusion
6|// 实际应用: MobileNet/EfficientNet 等轻量化视觉模型
7|//
8|// 参数: 1×4×4×1 输入, 3×3×1 卷积核, 1×4×4×1 输出
9|// 降级效果:  3 行输入 → affine (约 40 行) → LLVM (约 120 行)
10|
11|module {
12|  func.func @depthwise_conv(%input: memref<1x4x4x1xf32>,
13|                            %filter: memref<3x3x1xf32>,
14|                            %output: memref<1x4x4x1xf32>) {
15|    linalg.depthwise_conv_2d_nhwc_hwcm {
16|      dilations = dense<1> : tensor<2xi64>,
17|      strides = dense<1> : tensor<2xi64>
18|    } ins(%input, %filter : memref<1x4x4x1xf32>, memref<3x3x1xf32>)
19|      outs(%output : memref<1x4x4x1xf32>)
20|    return
21|  }
22|}
23|