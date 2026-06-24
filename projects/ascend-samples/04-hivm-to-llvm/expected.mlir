llvm.func @simple_add(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {
  %0 = llvm.load %A : !llvm.ptr -> vector<4xf32>
  %1 = llvm.load %B : !llvm.ptr -> vector<4xf32>
  %2 = llvm.fadd %0, %1 : vector<4xf32>
  llvm.store %2, %C : vector<4xf32>, !llvm.ptr
  llvm.return
}
