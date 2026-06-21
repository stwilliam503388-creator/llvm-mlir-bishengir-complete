// Max Pool — 最大池化 (下采样)
// 公式: 2x2窗口, stride=2, 取最大值, 4x4->2x2
// 一句话: 每4个像素取最亮的1个
// 专业角色: CNN下采样, 保留最强特征, 丢弃位置信息
// 用在哪: LeNet/AlexNet/VGG 下采样
// 降级: affine.for x4 + cmpf + select (手动循环)
// bishengir: linalg.pooling_nhwc_max (需自编译)

module {
  func.func @max_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
    %c0 = arith.constant 0.0 : f32
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        memref.store %c0, %output[%oh, %ow] : memref<2x2xf32>
        affine.for %kh = 0 to 2 {
          affine.for %kw = 0 to 2 {
            %v = affine.load %input[%oh * 2 + %kh, %ow * 2 + %kw] : memref<4x4xf32>
            %cur = memref.load %output[%oh, %ow] : memref<2x2xf32>
            %gt = arith.cmpf ogt, %v, %cur : f32
            %max = arith.select %gt, %v, %cur : f32
            memref.store %max, %output[%oh, %ow] : memref<2x2xf32>
          }
        }
      }
    }
    return
  }
}
