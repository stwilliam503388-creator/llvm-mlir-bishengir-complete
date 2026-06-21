// Global Avg Pool — 全局平均池化 (GAP)
// 公式: 整个特征图求平均, 4x4 -> 1
// 一句话: 把整张图压缩成1个数
// 专业角色: CNN 分类头前最后一层, 参数量0
// 用在哪: ResNet/MobileNet/GoogleNet 分类头
// 降级: affine.for x2 + 累加 + 平均因子
// bishengir: 组合 reduce + 除法

module {
  func.func @global_avg_pool(%input: memref<4x4xf32>, %output: memref<f32>) {
    %c16 = arith.constant 0.0625 : f32
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %output[] : memref<f32>
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        %v = affine.load %input[%i, %j] : memref<4x4xf32>
        %acc = memref.load %output[] : memref<f32>
        %sum = arith.addf %acc, %v : f32
        memref.store %sum, %output[] : memref<f32>
      }
    }
    %sum = memref.load %output[] : memref<f32>
    %gap = arith.mulf %sum, %c16 : f32
    memref.store %gap, %output[] : memref<f32>
    return
  }
}
