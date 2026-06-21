// ==- depthwise_conv_4x4.mlir - 深度可分离卷积 -==//
//
// 逐通道卷积（depthwise convolution），Mobilenet 的核心算子
// 对应 AscendNPU-IR: linalg.depthwise_conv_2d_nhwc_hwcm
// 等价 bishengir-opt: --convert-linalg-to-hfusion
// 实际应用: MobileNet/EfficientNet 等轻量化视觉模型
//
// 参数: 1×4×4×1 输入, 3×3×1 卷积核, 1×4×4×1 输出
// 降级效果:  3 行输入 → affine (约 40 行) → LLVM (约 120 行)

module {
  func.func @depthwise_conv(%input: memref<1x4x4x1xf32>,
                            %filter: memref<3x3x1xf32>,
                            %output: memref<1x4x4x1xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwcm {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : memref<1x4x4x1xf32>, memref<3x3x1xf32>)
      outs(%output : memref<1x4x4x1xf32>)
    return
  }
}
