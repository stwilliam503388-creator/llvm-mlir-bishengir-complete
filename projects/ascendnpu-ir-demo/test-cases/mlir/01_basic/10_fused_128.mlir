// Fused (原始版) — 算子融合演示 (对应 triton/01_basic/08_fused.py) ⭐
// 公式: C = A + B; D = C * A, 128 元素
// 一句话: 两步连续操作演示算子融合概念, bishengir 原始测试用例
// 专业角色: Kernel Fusion 概念演示, 与 08_fused.mlir 功能相同但采用原始命名
// 用在哪: 编译器优化概念演示
// 降级: 连续两次 linalg.generic
// bishengir: HFusion 算子融合概念演示
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.addf
// CHECK: arith.mulf
//
module {
  func.func @fused(%A: memref<128xf32>, %B: memref<128xf32>, %D: memref<128xf32>) {
    %C = memref.alloc() : memref<128xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : memref<128xf32>, memref<128xf32>)
      outs(%C : memref<128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%C, %A : memref<128xf32>, memref<128xf32>)
      outs(%D : memref<128xf32>) {
    ^bb0(%c: f32, %a: f32, %d: f32):
      %prod = arith.mulf %c, %a : f32
      linalg.yield %prod : f32
    }
    memref.dealloc %C : memref<128xf32>
    return
  }
}