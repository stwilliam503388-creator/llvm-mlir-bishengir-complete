// ==- max_pool_4x4.mlir - 最大池化 (affine.for 实现) -==//
// 2x2 kernel, stride=2, valid padding
// 输入 4x4 → 输出 2x2
// 用 affine.for 代替 linalg.pooling named op
// 原因: Homebrew LLVM 22 未编译 pooling named op
// 对应 AscendNPU-IR: bishengir 的 linalg.pooling_nhwc_max
// 降级: 9 行 → 80 行 LLVM

module {
  func.func @max_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
    %c0 = arith.constant 0.0 : f32
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
        %init = memref.load %output[%oh, %ow] : memref<2x2xf32>
        affine.for %kh = 0 to 2 {
          affine.for %kw = 0 to 2 {
            %v = affine.load %input[%oh * 2 + %kh, %ow * 2 + %kw] : memref<4x4xf32>
            %gt = arith.cmpf ogt, %v, %init : f32
            %max = arith.select %gt, %v, %init : f32
            memref.store %max, %output[%oh, %ow] : memref<2x2xf32>
          }
        }
      }
    }
    return
  }
}
