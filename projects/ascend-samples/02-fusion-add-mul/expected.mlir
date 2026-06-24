func.func @fused_add_mul(%A: tensor<4xf32>, %B: tensor<4xf32>, %C: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = husion.elemwise_binary "add_mul" ins(%A, %B, %C) outs(%A) : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
