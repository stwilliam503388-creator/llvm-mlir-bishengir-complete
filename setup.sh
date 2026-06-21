#!/bin/bash
# setup.sh — 检查本项目所需依赖
set -e

echo "╔════════════════════════════════════════════╗"
echo "║  llvm-mlir-bishengir: 环境检查             ║"
echo "╚════════════════════════════════════════════╝"

check() {
    if command -v "$1" &>/dev/null; then
        ver=$("$@" 2>&1 | head -1)
        echo "  ✅  $1  ($ver)"
    else
        echo "  ❌  $1  (未安装)"
        MISSING=1
    fi
}

MISSING=0

check mlir-opt
check mlir-tblgen
check cmake --version
check g++ --version
check python3 --version
check git --version

if [ -f /opt/homebrew/bin/ninja ]; then
    echo "  ✅  ninja  ($(ninja --version))"
else
    echo "  🟡  ninja  (可选，用于加速 CMake 构建)"
fi

echo ""
if [ $MISSING -eq 1 ]; then
    echo "❌ 部分依赖缺失，请安装:"
    echo "   brew install llvm cmake   # LLVM/MLIR + CMake"
else
    echo "✅ 所有核心依赖就绪"
fi
