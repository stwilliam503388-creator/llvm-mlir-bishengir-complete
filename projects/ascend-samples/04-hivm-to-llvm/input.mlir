// 用例 4 — hivm-to-llvm (hivm 版本：Ascend NPU 指令)
func.func @add(%A: memref<4xf16, #hivm.address_space<gm>>,
               %B: memref<4xf16, #hivm.address_space<gm>>,
               %C: memref<4xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ub_a = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%A) outs(%ub_a)
  %ub_b = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%B) outs(%ub_b)
  %ub_c = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_c)
  hivm.hir.store ins(%ub_c) outs(%C)
  return
}
