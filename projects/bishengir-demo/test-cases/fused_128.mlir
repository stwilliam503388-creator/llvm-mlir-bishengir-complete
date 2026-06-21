// ==- fused_128.mlir - 融合操作（bishengir demo）-==//
//
// 对应 AscendNPU-IR 源码:
//   本用例在官方测试中没有直接对应，演示 bishengir 的融合概念。
//   bishengir 中的融合等价:
//     linalg-fuse-elementwise-ops (若支持)
//     将连续两个 linalg.generic 合并为一个 hfusion.elemwise_binary
//
// C = A + B; D = C * A  →  融合后: D[i] = (A[i] + B[i]) * A[i]
// 融合减少一次内存读写，是 bishengir 的核心优化能力之一.
//===

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
