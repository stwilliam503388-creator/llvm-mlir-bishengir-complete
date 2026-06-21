// VecAdd — 向量加法
// 公式: C[i] = A[i] + B[i]
// 一句话: 两个数组对应位置逐元素相加
// 专业角色: 残差连接 (Residual Connection), y = x + F(x), 每层输出与输入直接相加
// 用在哪: Transformer 残差连接 / ResNet shortcut
// 降级: linalg.generic + arith.addf, 3行→38行LLVM (12.7×)
// bishengir: hfusion.elemwise_binary {fun = add}

// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.addf

module {
  func.func @vecadd(%A: memref<128xf16>, %B: memref<128xf16>, %C: memref<128xf16>) {
    linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]} ins(%A, %B : memref<128xf16>, memref<128xf16>) outs(%C : memref<128xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    }
    return
  }
}
