// Broadcast — 广播 (张量扩展)
// 公式: B[i][j] = A, 标量到矩阵
// 一句话: 把1个数复制到整个数组的每个位置
// 专业角色: 张量维度自动扩展, 用于 bias 加法 / 归一化参数广播
// 用在哪: 所有带 bias/归一化的层
// 降级: affine_map<(i,j) -> ()> (标量到矩阵映射)
// bishengir: hfusion.broadcast

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: affine.for

module {
  func.func @broadcast(%A: memref<f32>, %B: memref<4x4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(i,j) -> ()>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} ins(%A : memref<f32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
    }
    return
  }
}
