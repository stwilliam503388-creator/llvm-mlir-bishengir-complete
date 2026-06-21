// ==- avg_pool_4x4.mlir - 平均池化 (affine.for 实现) -==//
// 2x2 kernel, stride=2, valid padding
// 输入 4x4 → 输出 2x2
// 用 affine.for 代替 linalg.pooling named op
// 降级: 11 行 → 83 行 LLVM

module {
  func.func @avg_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
    %c04 = arith.constant 0.25 : f32
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        %c0 = arith.constant 0.0 : f32
        memref.store %c0, %output[%oh, %ow] : memref<2x2xf32>
        affine.for %kh = 0 to 2 {
          affine.for %kw = 0 to 2 {
            %v = affine.load %input[%oh * 2 + %kh, %ow * 2 + %kw] : memref<4x4xf32>
            %acc = memref.load %output[%oh, %ow] : memref<2x2xf32>
            %sum = arith.addf %v, %acc : f32
            memref.store %sum, %output[%oh, %ow] : memref<2x2xf32>
          }
        }
        %sum = memref.load %output[%oh, %ow] : memref<2x2xf32>
        %avg = arith.mulf %sum, %c04 : f32
        memref.store %avg, %output[%oh, %ow] : memref<2x2xf32>
      }
    }
    return
  }
}
