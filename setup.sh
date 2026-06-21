#!/bin/bash
# setup.sh — 检查本项目所需依赖
set -e

echo "╔════════════════════════════════════════════╗"
echo "║  ascend-npu-compiler-learning: 环境检查    ║"
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

check llvm-config --version
check clang++ --version
check cmake --version
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
    echo "   macOS: brew install llvm cmake"
    echo "   Linux: sudo apt install llvm-18-dev clang-18 cmake"
    echo ""
    echo "   详见: docs/llvm/00-环境搭建.md"
else
    echo "✅ 所有核心依赖就绪"
    echo ""
    echo "下一步: 从 docs/primer/ 开始学习，或直接进入"
    echo "        projects/hello-pass/ 跑你的第一个 Pass"
fi
