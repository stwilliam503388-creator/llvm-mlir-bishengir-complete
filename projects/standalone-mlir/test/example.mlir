// ==- example.mlir - Standalone dialect 测试 -==//
//
// 运行方式:
//   standalone-opt test/example.mlir
//   standalone-opt test/example.mlir --count-ops
//   standalone-opt test/example.mlir --elim-transpose
//   standalone-opt test/example.mlir --canonicalize
//==

module {
  func.func @main() {
    // 定义 2x3 矩阵常量
    %0 = standalone.constant { dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : !standalone.tensor<2x3>
    %1 = standalone.constant { dense<[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]> : tensor<2x3xf64> } : !standalone.tensor<2x3>

    // 加法
    %2 = standalone.add %0, %1 : !standalone.tensor<2x3>

    // 转置
    %3 = standalone.transpose(%2) : !standalone.tensor<2x3> -> !standalone.tensor<3x2>

    // 冗余转置 (transpose(transpose(x)) → should be eliminated)
    %4 = standalone.transpose(%3) : !standalone.tensor<3x2> -> !standalone.tensor<2x3>

    // 乘法
    %5 = standalone.mul %4, %0 : !standalone.tensor<2x3>

    // 打印
    standalone.print(%5) : !standalone.tensor<2x3>

    // 返回
    standalone.return
  }
}
