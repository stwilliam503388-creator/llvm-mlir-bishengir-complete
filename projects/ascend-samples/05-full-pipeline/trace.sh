#!/bin/bash
# 追踪一条 add 指令经过 AscendNPU-IR 的完整变换
# 需要 AscendNPU-IR 已构建

if [ -z "$ASCEND_BUILD" ]; then
  echo "设置 ASCEND_BUILD 为 AscendNPU-IR 的 build 目录："
  echo "  export ASCEND_BUILD=~/AscendNPU-IR/build"
  exit 1
fi

INPUT="$(cd "$(dirname "$0")" && pwd)/input.mlir"

echo "=== Step 0: 原始 linalg IR ==="
cat "$INPUT"
echo ""

echo "=== Step 1: linalg → hivm.hir ==="
"$ASCEND_BUILD/bin/bishengir-opt" \
  --pass-pipeline="builtin.module(convert-linalg-to-hivm)" \
  "$INPUT" 2>/dev/null || echo "(需要 AscendNPU-IR 已构建)"
echo ""

echo "=== Step 2: hivm.hir → llvm ==="
"$ASCEND_BUILD/bin/bishengir-opt" \
  --pass-pipeline="builtin.module(convert-hivm-to-llvm)" \
  "$INPUT" 2>/dev/null || echo "(需要 AscendNPU-IR 已构建)"
echo ""

echo "🎉 完整 Lowering 路径追踪完毕"
echo ""
echo "手动对比各步输出，或参考："
echo "  ascend-samples/01-simple-add/expected.mlir  (linalg → hivm)"
echo "  ascend-samples/04-hivm-to-llvm/expected.mlir  (hivm → llvm)"
