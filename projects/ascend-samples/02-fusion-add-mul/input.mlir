func.func @fused_add_mul(%A: tensor<4xf32>, %B: tensor<4xf32>, %C: tensor<4xf32>)
    -> tensor<4xf32> {
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<4xf32>, tensor<4xf32>)
    outs(%A : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.addf %a, %b : f32
    linalg.yield %0 : f32
  } -> tensor<4xf32>
  %mul = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%add, %C : tensor<4xf32>, tensor<4xf32>)
    outs(%add : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    linalg.yield %0 : f32
  } -> tensor<4xf32>
  func.return %mul : tensor<4xf32>
}
