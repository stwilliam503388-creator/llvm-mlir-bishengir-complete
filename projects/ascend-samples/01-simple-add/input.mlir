func.func @simple_add(%A: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<4xf32>, tensor<4xf32>)
    outs(%A : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
