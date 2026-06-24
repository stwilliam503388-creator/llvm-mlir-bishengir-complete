// 用例 3 — hivm 操作拆解 (linalg 版本)
func.func @hivm_add(%A: tensor<4xf16>, %B: tensor<4xf16>) -> tensor<4xf16> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<4xf16>, tensor<4xf16>)
    outs(%A : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %add = arith.addf %a, %b : f16
    linalg.yield %add : f16
  } -> tensor<4xf16>
  func.return %0 : tensor<4xf16>
}
