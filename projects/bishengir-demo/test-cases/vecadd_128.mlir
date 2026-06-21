// ==- vecadd_128.mlir - 向量加法（bishengir demo）-==//
// bishengir 流水线对照:
//   linalg.generic { arith.addf }
//   → -convert-linalg-to-hfusion  → hfusion.elemwise_binary {add}
//   → -convert-hfusion-to-hivm   → hivm.load + hivm.vadd + hivm.store
//
// 等价标准 MLIR 流水线（在当前环境可运行）:
//   --convert-linalg-to-affine-loops  → affine.for + arith.addf
//   --lower-affine                    → scf.for
//   --convert-scf-to-cf               → cf.br
//   --convert-func-to-llvm            → llvm.func + llvm.load/store
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
