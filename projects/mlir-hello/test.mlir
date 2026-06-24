func.func @add(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  func.return %sum : i32
}

func.func @say_hello() {
  func.return
}
