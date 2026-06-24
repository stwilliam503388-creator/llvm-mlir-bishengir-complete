// 用例 4 — hivm-to-llvm (LLVM 版本：ConvertHivmToLLVM 之后)
llvm.func @add(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {
  // alloc → alloca
  %ub_a = llvm.alloca %sz : !llvm.ptr
  %ub_b = llvm.alloca %sz : !llvm.ptr
  %ub_c = llvm.alloca %sz : !llvm.ptr
  // hivm.hir.load → llvm.load
  %va = llvm.load %A : !llvm.ptr -> vector<4xf16>
  llvm.store %va, %ub_a : vector<4xf16>, !llvm.ptr
  %vb = llvm.load %B : !llvm.ptr -> vector<4xf16>
  llvm.store %vb, %ub_b : vector<4xf16>, !llvm.ptr
  // hivm.hir.vadd → llvm.fadd
  %vc = llvm.fadd %va, %vb : vector<4xf16>
  llvm.store %vc, %ub_c : vector<4xf16>, !llvm.ptr
  // hivm.hir.store → llvm.store (to global)
  %vc_loaded = llvm.load %ub_c : !llvm.ptr -> vector<4xf16>
  llvm.store %vc_loaded, %C : vector<4xf16>, !llvm.ptr
  llvm.return
}
