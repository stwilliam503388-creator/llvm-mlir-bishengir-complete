// ==- matmul_4x4x4.mlir - 矩阵乘法（bishengir demo）-==//
// bishengir 中 linalg.matmul 会降低到 Cube 指令
// 标准 MLIR 等效: linalg.matmul → affine.for → llvm

module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
                 outs(%C : memref<4x4xf32>)
    return
  }
}
