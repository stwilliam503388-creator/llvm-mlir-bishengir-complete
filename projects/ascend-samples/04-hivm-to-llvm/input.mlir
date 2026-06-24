func.func @simple_add(%A: memref<4xf32>, %B: memref<4xf32>, %C: memref<4xf32>) {
  %va = hivm.load %A : memref<4xf32> -> vector<4xf32>
  %vb = hivm.load %B : memref<4xf32> -> vector<4xf32>
  %vc = hivm.vadd %va, %vb : vector<4xf32>
  hivm.store %vc, %C : memref<4xf32>
  func.return
}
