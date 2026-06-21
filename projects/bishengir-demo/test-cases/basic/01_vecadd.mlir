// ==- vecadd_128.mlir - 向量加法（bishengir demo）-==//
//
// 对应 AscendNPU-IR 源码:
//   输入格式: bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
//   Pass 实现: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
//   等价 bishengir-opt 命令:
//     bishengir-opt --convert-linalg-to-hfusion --convert-hfusion-to-hivm
//   预期 bishengir 输出: hfusion.elemwise_binary {fun = add}
//
// 本文件用标准 MLIR (mlir-opt) 模拟上述降级过程.
// 区别: 标准路径输出 affine.for + arith.addf, bishengir 输出 hivm.vadd.
//===

module {
  func.func @vecadd(%A: memref<128xf16>, %B: memref<128xf16>, %C: memref<128xf16>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : memref<128xf16>, memref<128xf16>)
      outs(%C : memref<128xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    }
    return
  }
}
