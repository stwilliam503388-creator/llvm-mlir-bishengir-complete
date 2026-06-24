#!/bin/bash
# 追踪一条 add 指令经过所有 Pass 的完整变换
# 需要 AscendNPU-IR 已构建

if [ -z "$ASCEND_BUILD" ]; then
  echo "设置 ASCEND_BUILD 为 AscendNPU-IR 的 build 目录："
  echo "  export ASCEND_BUILD=~/AscendNPU-IR/build"
  exit 1
fi

PIPELINE="builtin.module(
  convert-linalg-to-hfusion,
  convert-hfusion-to-hivm,
  convert-hivm-to-llvm
)"

INPUT="$(cd "$(dirname "$0")" && pwd)/input.mlir"

echo "=== Step 0: 原始 linalg IR ==="
cat "$INPUT"
echo ""

echo "=== Step 1: linalg → husion ==="
"$ASCEND_BUILD/bin/ascendnpu-ir-opt" \
  --pass-pipeline="builtin.module(convert-linalg-to-hfusion)" \
  "$INPUT" 2>/dev/null
echo ""

echo "=== Step 2: husion → hivm ==="
"$ASCEND_BUILD/bin/ascendnpu-ir-opt" \
  --pass-pipeline="builtin.module(convert-hfusion-to-hivm)" \
  "$INPUT" 2>/dev/null
echo ""

echo "=== Step 3: hivm → llvm ==="
"$ASCEND_BUILD/bin/ascendnpu-ir-opt" \
  --pass-pipeline="$PIPELINE" \
  "$INPUT" 2>/dev/null
echo ""

echo "🎉 完整 Lowering 路径追踪完毕"
