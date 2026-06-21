#!/bin/bash
# variant3_hw_mapping.sh — 硬件指令映射模拟
# 原理: 用 func.call 代替完整展开，模拟 bishengir 的 hivm.mmul
# bishengir 实际做法:
#   linalg.matmul → hfusion.cube_matmul → hivm.mmul (1行NPU指令)
# 本 demo 模拟: 保留 matmul 语义，不展开到标量
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"

cat > /tmp/matmul_hw.mlir << 'MLIR'
// ==- matmul_hw.mlir - 硬件映射模拟 -==//
// 模拟 bishengir 的 hivm.mmul:
//   在完整降级前插入 func.call，
//   表示该操作由硬件指令（Cube 单元）直接执行。
//===

module {
  // 声明硬件加速的矩阵乘函数
  func.func private @mmul_4x4xf32(memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)

  func.func @matmul_hw(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    // 调用硬件矩阵乘指令 —— 等价于 bishengir 的 hivm.mmul
    func.call @mmul_4x4xf32(%A, %B, %C) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    return
  }
}
MLIR

echo "═══ Variant 3: 硬件指令映射模拟 (hivm.mmul) ═══"
echo "输入:"
echo "  func.call @mmul_4x4xf32  ← 等价于 hivm.mmul"
echo ""

# 完整降级（保持 func.call 不被展开）
mlir-opt \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  /tmp/matmul_hw.mlir 2>&1 | tee "$RESULT_DIR/variant3_hw_llvm.mlir" | wc -l
echo "行 (LLVM, 保持 call 不展开)"
