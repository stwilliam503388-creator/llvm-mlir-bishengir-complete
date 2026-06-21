// ==- fused_128.mlir - 融合操作（bishengir demo）-==//
// C = A + B; D = C * A → 可融合为一轮 kernel

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
