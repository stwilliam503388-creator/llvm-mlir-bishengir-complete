// ==- global_avg_pool_4x4.mlir - 全局平均池化 -==//
// 将 4x4 特征图全局平均为 1 个标量
// 用途: ResNet/MobileNet 分类头前的最后一层
// 降级: 9 行 → 52 行 LLVM

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
