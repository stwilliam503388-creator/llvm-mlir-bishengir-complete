// 用例 2 — fusion-add-mul (linalg：两个独立操作)
func.func @fused_add_mul(%A: tensor<4xf16>, %B: tensor<4xf16>, %D: tensor<4xf16>)
    -> tensor<4xf16> {
  // 操作 1: add
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<4xf16>, tensor<4xf16>)
    outs(%A : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %0 = arith.addf %a, %b : f16
    linalg.yield %0 : f16
  } -> tensor<4xf16>
  // 操作 2: mul
  %mul = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%add, %D : tensor<4xf16>, tensor<4xf16>)
    outs(%add : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %0 = arith.mulf %a, %b : f16
    linalg.yield %0 : f16
  } -> tensor<4xf16>
  func.return %mul : tensor<4xf16>
}
