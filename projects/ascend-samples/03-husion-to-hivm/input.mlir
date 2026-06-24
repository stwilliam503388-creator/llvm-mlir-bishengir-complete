func.func @simple_add(%A: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
  %0 = husion.elemwise_binary "add" ins(%A, %B) outs(%A) : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
