// 用例 1 — simple-add (hivm 版本：ConvertLinalgToHivm 之后)
// 来源：bishengir/test/Integration/HIVM/VecAdd/add.mlir
func.func @simple_add(%A: memref<4xf16, #hivm.address_space<gm>>,
                      %B: memref<4xf16, #hivm.address_space<gm>>,
                      %C: memref<4xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ub_a = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%A : memref<4xf16, #hivm.address_space<gm>>)
                outs(%ub_a : memref<4xf16, #hivm.address_space<ub>>)
  %ub_b = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%B : memref<4xf16, #hivm.address_space<gm>>)
                outs(%ub_b : memref<4xf16, #hivm.address_space<ub>>)
  %ub_c = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b : memref<4xf16, #hivm.address_space<ub>>,
                                memref<4xf16, #hivm.address_space<ub>>)
                outs(%ub_c : memref<4xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%ub_c : memref<4xf16, #hivm.address_space<ub>>)
                 outs(%C : memref<4xf16, #hivm.address_space<gm>>)
  return
}
