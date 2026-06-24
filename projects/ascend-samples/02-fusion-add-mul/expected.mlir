// 用例 2 — fusion-add-mul (hivm：融合后共用 Unified Buffer)
func.func @fused_add_mul(%A: memref<4xf16, #hivm.address_space<gm>>,
                          %B: memref<4xf16, #hivm.address_space<gm>>,
                          %D: memref<4xf16, #hivm.address_space<gm>>,
                          %C: memref<4xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ub_a = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%A) outs(%ub_a)
  %ub_b = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%B) outs(%ub_b)
  // add + mul 融合：中间结果留在 Unified Buffer，不写回 HBM
  %ub_tmp = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_tmp)
  %ub_d = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%D) outs(%ub_d)
  %ub_c = memref.alloc() : memref<4xf16, #hivm.address_space<ub>>
  hivm.hir.vmul ins(%ub_tmp, %ub_d) outs(%ub_c)
  hivm.hir.store ins(%ub_c) outs(%C)
  return
}
