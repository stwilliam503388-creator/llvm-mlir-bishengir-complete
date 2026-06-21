// ==- broadcast_4x4.mlir - 广播标量到矩阵 -==//
//
// B[i][j] = A — 把标量 A 广播到 4x4 矩阵
// 对应 AscendNPU-IR: hfusion.broadcast
// 常见的实际场景: bias 广播 (add bias to matmul output)
// Pass 参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
//
// MLIR 关键概念: affine_map<(i,j) -> ()>
//   "()" 表示输入是标量，与输出 (i,j) 的每个元素都相同
//   这比手动写循环更清晰地表达了广播语义

module {
  func.func @broadcast(%A: memref<f32>, %B: memref<4x4xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j) -> ()>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<f32>) outs(%B : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
    }
    return
  }
}
