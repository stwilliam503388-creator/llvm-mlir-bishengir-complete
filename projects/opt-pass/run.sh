#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_PREFIX="/opt/homebrew/opt/llvm"
export PATH="$LLVM_PREFIX/bin:$PATH"

if ! command -v llvm-config &>/dev/null; then
  echo "❌ 找不到 llvm-config。brew install llvm"
  exit 1
fi

echo "🔨 构建 OptPass（死代码消除）..."

CXXFLAGS="-std=c++17 -I$LLVM_PREFIX/include"

$LLVM_PREFIX/bin/clang++ $CXXFLAGS \
  OptPass.cpp -shared -fPIC \
  -L"$LLVM_PREFIX/lib" -lLLVM \
  -Wl,-rpath,"$LLVM_PREFIX/lib" \
  -o libOptPass.dylib

echo "✅ 构建成功！运行测试："
echo ""
echo "--- 优化前 ---"
cat test.ll
echo ""
echo "--- 优化后 ---"
$LLVM_PREFIX/bin/opt -load-pass-plugin ./libOptPass.dylib \
  --passes="opt-pass" test.ll -S 2>&1
echo ""
echo "--- 预期输出 ---"
cat test_expected.ll
echo ""
echo "🎉 OptPass 运行成功！对比"优化后"和"预期输出"应一致"
