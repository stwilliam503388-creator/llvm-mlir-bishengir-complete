#!/bin/bash
set -e

# macOS: Homebrew keg-only LLVM 路径
LLVM_PREFIX="/opt/homebrew/opt/llvm"

if [ "$(uname)" = "Linux" ]; then
  LLVM_PREFIX="/usr/lib/llvm-18"
fi

export PATH="$LLVM_PREFIX/bin:$PATH"

if ! command -v llvm-config &>/dev/null; then
  echo "❌ 找不到 llvm-config。请先安装 LLVM："
  echo "   macOS: brew install llvm"
  echo "   Linux: sudo apt install llvm-18-dev"
  exit 1
fi

echo "🔨 构建 HelloPass（LLVM $(llvm-config --version)）..."
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH="$(llvm-config --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo ""
echo "✅ 构建成功！运行测试："
echo ""

# cmake MODULE 库在 macOS 产 .so，在 Linux 产 .so
LIB=$(ls libHelloPass.* 2>/dev/null | head -1)

opt -load-pass-plugin "$PWD/$LIB" \
    --passes="hello" \
    ../test.ll \
    -disable-output

echo ""
echo "---"
echo "🎉 HelloPass 运行成功！"
echo ""
echo "提示: 修改 HelloPass.cpp 后，重新运行 ./run.sh 即可"
