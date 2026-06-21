// Avg Pool — 平均池化 (下采样)
// 公式: 2x2窗口, stride=2, 取平均值, 4x4->2x2
// 一句话: 取窗口平均值而非最大值
// 专业角色: CNN下采样, 比 max pool 更平滑
// 用在哪: ResNet 分类头 / 平滑下采样
// 降级: affine.for x4 + 累加 + 除法
// bishengir: linalg.pooling_nchw_sum + 除法 (需自编译)
// RUN: mlir-opt --lower-affine %s | FileCheck %s
// RUN: mlir-opt --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK-NOT: affine.for
// CHECK: scf.for
// CHECK: arith.addf
// CHECK: arith.mulf

module {
  func.func @avg_pool(%input: memref<4x4xf32>, %output: memref<2x2xf32>) {
    %c04 = arith.constant 0.25 : f32
    %c0 = arith.constant 0.0 : f32
    affine.for %oh = 0 to 2 {
      affine.for %ow = 0 to 2 {
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
