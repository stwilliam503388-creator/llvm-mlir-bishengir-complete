// ==- matmul_4x4x4.mlir - 矩阵乘法（bishengir demo）-==//
//
// 对应 AscendNPU-IR 源码:
//   输入格式: bishengir/test/Conversion/LinalgToHFusion/matmul-to-hfusion.mlir
//   Pass 实现: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
//   等价 bishengir-opt 命令:
//     bishengir-opt --convert-linalg-to-hfusion
//   预期 bishengir 输出: hfusion.cube_matmul
//   （bishengir 保持 1 行语义，本 demo 展开为 74 行）
//
// bishengir 保留 hfusion.cube_matmul 并直接映射到 Ascend NPU Cube 硬件单元,
// 无需展开到标量指令。本文件的标准降级展示了"如果不保留语义"会怎样.
//===

module {
  func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
                 outs(%C : memref<4x4xf32>)
    return
  }
}
