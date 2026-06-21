# LLVM → MLIR → bishengir 完整学习项目

> 从 LLVM IR 入门到 bishengir（Ascend NPU）实战的全链路知识库与工程合集。
> MacBook Air (Apple Silicon) + LLVM 22.1.6 (Homebrew)

---

## 项目结构

```
llvm-mlir-bishengir-complete/
├── README.md                         ← 本文件（项目总览）
├── SUMMARY.md                        ← 本次 session 完整输出总结
│
├── docs/                             ← 知识库（15 篇笔记）
│   ├── llvm/                         ← LLVM 速通（7 篇）
│   │   ├── LLVM-L00-速通总览.md
│   │   ├── LLVM-L01-架构与HelloWorld.md
│   │   ├── LLVM-L02-类型系统与GEP.md
│   │   ├── LLVM-L03-控制流与Phi节点.md
│   │   ├── LLVM-L04-内置函数与属性.md
│   │   ├── LLVM-L05-Pass开发.md
│   │   └── LLVM-L06-IR速查表.md
│   │
│   └── mlir/                         ← MLIR 体系（8 篇，本期新增）
│       ├── MLIR-L00-速通与bishengir实战          — MLIR 核心 + bishengir 对照
│       ├── MLIR-L01-ToyTutorial速通-Ch1-Ch2      — Toy 语言 + Dialect 定义
│       ├── MLIR-L02-ToyTutorial速通-Ch3-Ch6      — Pattern + Partial Lowering
│       ├── MLIR-L03-自定义bishengirPass实战       — OpCounter + PeelTranspose
│       ├── MLIR-L04-Standalone实战总结            — cmake/Makefile + LLVM 22 适配
│       ├── MLIR-L05-ToyMini从零实现               — 纯 C++17 解析器
│       ├── MLIR-L06-TritonMLIR体系分析            — TT/TritonGPU 双 Dialect
│       ├── MLIR-L07-triton-ascend后端深度分析     — Python ↔ C++ 全链路
│       └── MLIR-L08-bishengir-demo可运行流水线    — mlir-opt 实操验证
│
├── projects/                         ← 工程项目（4 个）
│   ├── bishengir-demo/               ★ 可运行 MLIR 降级流水线（核心交付）
│   │   ├── test-cases/
│   │   │   ├── vecadd_128.mlir       — 向量加法 ✅ mlir-opt 验证通过
│   │   │   ├── matmul_4x4x4.mlir     — 矩阵乘法 ✅
│   │   │   └── fused_128.mlir        — 融合操作 ✅
│   │   ├── bishengir-demo.py         — Python 用例生成器
│   │   ├── run-demo.sh               — 批量运行脚本
│   │   └── README.md                 — 使用说明 + bishengir 对照表
│   │
│   ├── toy-mini/                     ★ 从零写 Toy 解析器
│   │   └── toymini.cpp               — 单文件 1412 行，g++ -std=c++17 编译通过
│   │
│   ├── standalone-mlir/              ★ 从零构建 MLIR dialect
│   │   ├── CMakeLists.txt
│   │   ├── Makefile
│   │   ├── include/standalone/
│   │   │   ├── StandaloneOps.td      — TableGen 定义（6 ops）
│   │   │   └── StandaloneDialect.h
│   │   ├── tools/
│   │   │   └── standalone-opt.cpp    — 全合一入口
│   │   └── test/example.mlir
│   │
│   └── bishengir-op-counter/         ★ 自定义 MLIR Pass 参考代码
│       ├── BishengirOpCounter.cpp    — 分析 Pass（walk 统计 ops）
│       └── BishengirPeelTranspose.cpp— 转换 Pass（冗余消除）
│
└── references/                       ← 参考源码
    └── triton-ascend/                — triton-ascend 源码摘录位置
```

---

## 核心学习路径

```
LLVM IR 基础                MLIR 概念                  bishengir 实战
────────────               ──────────                 ──────────────
L01 架构                   L00 MLIR 核心              → bishengir-demo
L02 类型系统/GEP           L01 Toy Dialect 定义        （3 用例可运行）
L03 控制流/Phi             L02 Pattern + Lowering      → bishengir-op-counter
L04 内置函数               L03 自定义 Pass              （分析+转换 Pass）
L05 Pass 开发              L04 Standalone 实战         → ascendnpu-ir 源码
L06 速查表                 L05 Toy Mini 手写分析
                           L06-L07 Triton 体系
                           L08 bishengir-demo
```

---

## 验证状态

| 能力 | 方式 | 结果 |
|------|------|------|
| MLIR 降级流水线 | `mlir-opt --convert-linalg-to-affine-loops ... --convert-func-to-llvm` | ✅ 3 用例全通过 |
| Toy 解析器 | `g++ -std=c++17 -o toymini toymini.cpp` | ✅ 编译 0 errors |
| TableGen | `mlir-tblgen --gen-op-decls StandaloneOps.td` | ✅ 语法正确 |
| CMake + MLIR | `cmake -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir` | ✅ 配置成功 |
| bishengir 对照 | 三阶段降级（Linalg→HFusion→HIVM）源码逐行解读 | ✅ 完成 |

---

## 快速开始

```bash
# 1. bishengir-demo（开箱即用）
cd projects/bishengir-demo
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir

# 2. Toy Mini（编译运行）
cd projects/toy-mini
g++ -std=c++17 -o toymini toymini.cpp && ./toymini

# 3. Standalone MLIR（编译）
cd projects/standalone-mlir
make -C build
```

---

## 依赖

- **macOS** (Apple Silicon)
- **LLVM 22.1.6** (Homebrew): `brew install llvm`
- **CMake ≥ 3.20**（可选，用于 standalone-mlir）
- **g++ / clang++** 支持 C++17
