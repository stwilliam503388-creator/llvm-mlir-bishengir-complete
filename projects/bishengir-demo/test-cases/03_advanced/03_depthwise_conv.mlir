// Depthwise Conv — 深度可分离卷积
// 公式: 逐通道3x3卷积, 计算量=标准卷积的1/9~1/3
// 一句话: 每个通道单独做卷积, 轻量化
// 专业角色: MobileNet/EfficientNet 核心算子
// 用在哪: MobileNet/EfficientNet
// 降级: linalg.depthwise_conv_2d_nhwc_hwcm
// bishengir: named op 直接映射

module {
  func.func @depthwise_conv(%input: memref<1x4x4x1xf32>, %filter: memref<3x3x1xf32>, %output: memref<1x4x4x1xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter : memref<1x4x4x1xf32>, memref<3x3x1xf32>) outs(%output : memref<1x4x4x1xf32>)
    return
  }
}
