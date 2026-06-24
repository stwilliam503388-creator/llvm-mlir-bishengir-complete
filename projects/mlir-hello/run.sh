#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_PREFIX="/opt/homebrew/opt/llvm"
LLVM_INC="/opt/homebrew/Cellar/llvm/22.1.6/include"
LLVM_LIB="$LLVM_PREFIX/lib"

if [ "$(uname)" = "Linux" ]; then
  LLVM_PREFIX="/usr/lib/llvm-18"
  LLVM_INC="$LLVM_PREFIX/include"
  LLVM_LIB="$LLVM_PREFIX/lib"
fi

export PATH="$LLVM_PREFIX/bin:$PATH"

echo "🔨 构建 hello-mlir（LLVM $(llvm-config --version)）..."

$LLVM_PREFIX/bin/clang++ -std=c++17 \
  -I"$LLVM_INC" \
  hello-mlir.cpp \
  -L"$LLVM_LIB" \
  -lMLIR -lLLVM \
  -Wl,-rpath,"$LLVM_LIB" \
  -o hello-mlir

echo "✅ 构建成功！运行测试："
echo ""
./hello-mlir test.mlir
echo ""
echo "---"
echo "🎉 HelloMLIRPass 运行成功！"
