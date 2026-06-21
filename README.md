# LLVM → MLIR → bishengir 完整学习项目

> 从 LLVM IR 入门到 bishengir（Ascend NPU）实战的全链路知识库与工程合集。
> MacBook Air (Apple Silicon) + LLVM 22.1.6 (Homebrew)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 项目结构

```
llvm-mlir-bishengir-complete/
├── README.md                     ← 本文件（项目总览）
├── SUMMARY.md                    ← 完整输出总结文档
├── LICENSE                       ← MIT 许可证
├── setup.sh                      ← 依赖检查脚本
│
├── docs/                         ← 知识库（15 篇笔记）
│   ├── llvm/                     ← LLVM 速通（7 篇）
│   │   ├── L00-速通总览.md       — 学习路线图
│   │   ├── L01-架构与HelloWorld  — 三段式架构、SSA
│   │   ├── L02-类型系统与GEP     — 类型系统、GEP 剥洋葱
│   │   ├── L03-控制流与Phi节点   — br 指令、φ 汇合
│   │   ├── L04-内置函数与属性    — memcpy、expect
│   │   ├── L05-Pass开发          — New PM、BBCounter
│   │   └── L06-IR速查表          — 指令、调试命令
│   │
│   └── mlir/                    ← MLIR 体系（8 篇）
│       ├── L00-速通与bishengir实战     — MLIR 核心概念 + bishengir 对照
│       ├── L01-ToyTutorial速通-Ch1-Ch2 — Toy 语言 + Dialect 定义
│       ├── L02-ToyTutorial速通-Ch3-Ch6 — Pattern + Partial Lowering
│       ├── L03-自定义bishengirPass     — OpCounter + PeelTranspose
│       ├── L04-Standalone实战           — CMake/Makefile + LLVM22 适配
│       ├── L05-ToyMini从零实现          — 纯 C++17 解析器
│       ├── L06-TritonMLIR体系           — TT/TritonGPU 双 Dialect
│       ├── L07-triton-ascend后端分析    — Python ↔ C++ 全链路
│       └── L08-bishengir-demo           — 可运行 mlir-opt 流水线
│
├── projects/                    ← 工程项目（4 个）
│   ├── bishengir-demo/          ★ 可运行降级流水线
│   │   ├── test-cases/           — 3 个 MLIR 用例（✅ 全部通过）
│   │   ├── bishengir-demo.py     — Python 用例生成器
│   │   ├── run-demo.sh           — 批量运行脚本
│   │   └── README.md             — 使用说明 + bishengir 对照
│   │
│   ├── toy-mini/                ★ 从零写 Toy 解析器
│   │   ├── toymini.cpp           — 1412 行，零依赖 ✅ 编译通过
│   │   └── README.md             — 语法说明 + 输出示例
│   │
│   ├── standalone-mlir/         ★ 从零构建 MLIR dialect
│   │   ├── CMakeLists.txt        — CMake 构建
│   │   ├── Makefile              — 备选 Makefile
│   │   ├── include/standalone/   — TableGen 定义（6 ops）
│   │   ├── tools/standalone-opt  — 全合一入口（2 passes）
│   │   ├── test/example.mlir     — 测试输入
│   │   └── README.md             — 构建说明
│   │
│   └── bishengir-op-counter/    ★ 自定义 Pass 参考代码
│       ├── BishengirOpCounter.cpp    — 分析 Pass（walk 统计）
│       ├── BishengirPeelTranspose.cpp— 转换 Pass（冗余消除）
│       └── README.md                 — 编译说明 + 关键模式
│
└── references/                   ← 外部源码索引
    └── README.md                  — triton-ascend 核心文件位置
```

---

## 学习路径

```
LLVM IR 基础                MLIR 概念                  bishengir 实战
────────────               ──────────                 ──────────────
L01 架构 (SSA)             L00 MLIR 核心               → bishengir-demo
L02 类型系统/GEP           L01 Toy Dialect 定义         （3 用例可运行）
L03 控制流/Phi             L02 Pattern + Lowering      → bishengir-op-counter
L04 内置函数               L03 自定义 Pass              （分析 + 转换 Pass）
L05 Pass 开发              L04 Standalone 实战         → ascendnpu-ir 源码
L06 速查表                 L05 Toy Mini 手写分析
                           L06-L07 Triton 体系
                           L08 bishengir-demo
```

**笔记 ↔ 项目对照**：

| 项目 | 对应笔记 | 知识点 |
|------|---------|--------|
| bishengir-demo | L00, L08 | 三段降级、mlir-opt 流水线 |
| toy-mini | L01, L05 | Lexer/Parser/AST、MLIRGen |
| standalone-mlir | L03, L04 | TableGen、Pass 注册、CMake |
| bishengir-op-counter | L02, L03 | Pattern Rewriting、ConversionTarget |

---

## 验证状态

| 能力 | 方式 | 结果 | 对应笔记 |
|------|------|------|---------|
| MLIR 降级流水线 | `mlir-opt --convert-linalg-to-affine-loops ...` | ✅ 3 用例全通过 | L08 |
| Toy 解析器 | `g++ -std=c++17 -o toymini toymini.cpp` | ✅ 编译 0 errors | L05 |
| TableGen | `mlir-tblgen --gen-op-decls StandaloneOps.td` | ✅ 语法正确 | L04 |
| CMake + MLIR | `cmake -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir` | ✅ 配置成功 | L04 |
| bishengir 三阶段降级 | Linalg→HFusion→HIVM 源码逐行解读 | ✅ 完成 | L00 |
| Triton MLIR 体系 | triton-ascend 源码分析 | ✅ 完成 | L06, L07 |
| 三项目对照 | Toy ↔ bishengir ↔ Triton | ✅ 完成 | L00, L06 |

---

## 快速开始

```bash
# 1. 检查环境
bash setup.sh

# 2. bishengir-demo（开箱即用）
cd projects/bishengir-demo
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
         --convert-scf-to-cf --convert-func-to-llvm \
         test-cases/vecadd_128.mlir

# 3. Toy Mini（编译运行）
cd projects/toy-mini
g++ -std=c++17 -o toymini toymini.cpp && ./toymini

# 4. Standalone MLIR（编译）
cd projects/standalone-mlir
export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -C build
```

---

## 依赖

| 工具 | 版本 | 安装 | 用于 |
|------|------|------|------|
| LLVM/MLIR | 22.1.6 | `brew install llvm` | mlir-opt, mlir-tblgen |
| cmake | ≥ 3.20 | `brew install cmake` | standalone-mlir |
| g++ / clang++ | C++17 | `xcode-select --install` | toy-mini |
| ninja | (可选) | `brew install ninja` | 加速 CMake 构建 |
| python3 | ≥ 3.8 | (系统自带) | bishengir-demo.py |

---

## 关键技术对照

### bishengir ↔ 标准 MLIR

```
bishengir:                  标准 MLIR (本项目):
linalg.generic              linalg.generic
  ↓ -convert-linalg-to-hfusion  ↓ --convert-linalg-to-affine-loops
hfusion.elemwise_binary     affine.for + arith.addf
  ↓ -convert-hfusion-to-hivm   ↓ --lower-affine --scf-to-cf --func-to-llvm
hivm.load/vadd/store        llvm.load + llvm.add + llvm.store
```

### 三项目对照

| 维度 | Toy Tutorial | bishengir | Triton |
|------|-------------|----------|--------|
| **编程模型** | Toy 语言 | MLIR (Linalg) | Triton Python kernel |
| **高级 IR** | toy ops | linalg.generic | tt.load/dot/store |
| **中间 IR** | affine.for | hfusion.elemwise_binary | TritonGPU (layout) |
| **低级 IR** | scf + memref | hivm.vadd/load/store | LLVM IR / AIR |
| **目标** | CPU (LLVM JIT) | Ascend NPU | CPU / GPU / Ascend |
| **Dialect 数** | 1 | 8 | 2 (TT + TritonGPU) |
