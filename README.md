# LLVM → MLIR → bishengir: Ascend NPU 编译器全链路学习

> 从 LLVM IR 入门到 MLIR Dialect 开发，最终对接 bishengir (Ascend NPU) 的完整学习路径与工程合集

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM-22.1.6-blue)](https://llvm.org)
[![macOS](https://img.shields.io/badge/macOS-26.5.1-ff69b4)](https://www.apple.com/macos)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M5-green)]()

---

## 一、项目背景

### 为什么有这个项目

随着 AI 芯片（Ascend NPU）的普及，了解编译器技术栈成为 AI 工程师的核心能力。本项目源于一条具体的学习主线：

```
Triton Python kernel  →  Triton IR (TT Dialect)  →  bishengir (Ascend NPU)
```

编译器涉及的技术栈从 **LLVM IR 基础** 到 **MLIR dialect 定义** 再到 **Ascend 专用 IR (HFusion / HIVM)**，跨度大、难点多。现有教程要么偏学术（LLVM 源码），要么偏应用（只讲 Triton 使用），缺乏一条从零到 bishengir 的动手路径。

本项目填补了这个空白。

### 覆盖范围

```
基础知识 ←───────── 核心概念 ←────────── 工程实践
─────────           ─────────           ──────────
LLVM IR            MLIR Dialect        bishengir-demo
  SSA 形式            dialect 定义        可运行降级流水线
  类型系统/GEP        Operation/Region    Linalg→affine→LLVM
  控制流/Phi          Pattern Rewriting   bishengir 三阶段对照
  Pass 开发           Dialect Conversion  向量加法 / 矩阵乘法 / 融合
                    Pass 管理器          自定义 MLIR Pass
                    TableGen ODS         OpCounter / PeelTranspose
                    mlir-opt 工具链       Toy Mini 解析器
                                         纯 C++17 零依赖
                                         Standalone MLIR 项目
                                         从零构建 dialect
                                         CMake + TableGen
                                         Triton → Ascend 对接
                                         triton-ascend 后端源码分析
```

### 适用读者

- 想理解 **MLIR/JAX/Ascend 编译栈** 的 AI 工程师
- 需要开发 **自定义 MLIR dialect** 的编译器开发者
- 阅读 **Triton 源码** 时遇到 MLIR 瓶颈的学习者
- 熟悉 PyTorch/Triton 使用，但想深入底层的工作者

### 前置知识

| 要求 | 说明 |
|------|------|
| ✅ C++ 基础 | 能读 C++17 代码 |
| ✅ 编译器直觉 | 了解 AST、IR 概念 |
| 🟡 LLVM 经验 | 没有也没关系，笔记从零开始 |
| ❌ Ascend NPU | 不需要硬件，所有验证在 CPU 上完成 |

---

## 二、项目总览

### 56 个文件，覆盖 4 个层次

```
层次 1: 文档 (15 篇笔记, ~76KB)
├── LLVM IR 基础 (7 篇)     — 从 SSA 到 Pass 开发
└── MLIR 体系 (8 篇)        — 从 dialect 概念到 Triton 对接

层次 2: 可运行工程 (4 个项目)
├── bishengir-demo ★        — 3 个 MLIR 用例，mlir-opt 验证通过
├── toy-mini                 — 纯 C++17 Toy 解析器，编译通过
├── standalone-mlir          — CMake + Makefile + TableGen 自建 dialect
└── bishengir-op-counter     — 分析 + 转换 Pass 参考代码

层次 3: 设施
├── setup.sh                 — 依赖检查
├── LICENSE                  — MIT 开源
├── .gitattributes           — 换行符管理
└── references/              — 外部源码索引

层次 4: 外部源码（不在本仓库）
├── ascendnpu-ir (bishengir) — Ascend NPU MLIR 转换 Pass
└── triton-ascend            — Triton 前端对接
```

### 已验证

| 验证项 | 结果 | 说明 |
|--------|------|------|
| `mlir-opt` 降级流水线 | ✅ 3/3 用例通过 | vecadd / matmul / fused |
| `g++ -std=c++17` 编译 | ✅ 0 errors | toymini.cpp (1,412 行) |
| `mlir-tblgen` TableGen | ✅ 语法通过 | StandaloneOps.td (6 ops) |
| CMake + MLIR 集成 | ✅ 配置成功 | 跳过 AddMLIR 冲突 |
| bishengir 源码分析 | ✅ 完成 | 3 个 Conversion Pass 逐行解读 |
| Triton MLIR 体系 | ✅ 完成 | TT / TritonGPU 双 Dialect 分析 |

---

## 三、学习路径

### Stage 0: LLVM IR 基础（约 3 天）

目标：理解 LLVM 编译器的核心模型，为 MLIR 打下基础。

```
笔记路径: docs/llvm/
验证方式: 读懂 .ll 文件 + 理解 Pass 结构
```

| 步骤 | 笔记 | 知识点 | 产出 |
|------|------|--------|------|
| 0.1 | L00 速通总览 | 三段式架构、学习路线图 | 整体认知 |
| 0.2 | L01 架构与 HelloWorld | SSA 形式、Module/Func 结构 | 能读 `.ll` 文件 |
| 0.3 | L02 类型系统与 GEP | `iN` 类型、类型转换、GEP 剥洋葱 | 理解地址计算 |
| 0.4 | L03 控制流与 Phi | `br` 指令、φ 节点汇合规则 | 理解 CFG |
| 0.5 | L04 内置函数与属性 | `llvm.memcpy`、`expect` 内建 | 了解优化基础 |
| 0.6 | L05 Pass 开发 | New PM 架构、FunctionPass 骨架 | 能写 BBCounter |
| 0.7 | L06 IR 速查表 | 常用指令、调试命令、FileCheck | 快速参考 |

**关键突破**: 理解 SSA + φ 节点。这是 MLIR 的 Region 概念的基础。

### Stage 1: MLIR 核心概念（约 5 天）

目标：掌握 MLIR 的多层 IR 哲学，理解 dialect / operation / pass 三大概念。

```
笔记路径: docs/mlir/L00 ~ L04
验证方式: 运行 bishengir-demo + 读懂 standalone-mlir
```

| 步骤 | 笔记 | 知识点 | 对应项目 |
|------|------|--------|---------|
| 1.1 | L00 速通与 bishengir | dialect/region/operation 概念 | → bishengir-demo |
| 1.2 | L01 Toy Ch1-2 | TableGen 语法、Ops.td 结构 | → toy-mini |
| 1.3 | L02 Toy Ch3-6 | Pattern Rewriting、ConversionTarget | → bishengir-op-counter |
| 1.4 | L03 自定义 Pass | walk / OpRewritePattern 两种模式 | → bishengir-op-counter |
| 1.5 | L04 Standalone 实战 | CMake + Makefile + LLVM 22 适配 | → standalone-mlir |

**关键突破**: 理解 MLIR 的 **多层 IR 概念**——为什么需要多个 dialect，如何用 Pass 做 dialect 转换。

### Stage 2: 工程实战（约 3 天）

目标：从读到写，产出可运行的 MLIR 工程。

| 步骤 | 项目 | 行动 | 验证 |
|------|------|------|------|
| 2.1 | bishengir-demo | 运行 3 个用例，观察降级过程 | `mlir-opt` 输出 |
| 2.2 | bishengir-demo | 修改参数重新生成，理解 IR 膨胀 | 对比 stage0→stage3 |
| 2.3 | toy-mini | 编译运行，修改语法扩展 | `./toymini` 输出 |
| 2.4 | standalone-mlir | 编译，跑自定义 Pass | `--count-ops` |
| 2.5 | bishengir-op-counter | 阅读源码，理解模式 | 对照 Toy Tutorial |

**关键突破**: 能用 `mlir-opt` 验证自己的 dialect 理解。

### Stage 3: 体系对照（约 2 天）

目标：将学到的知识对标到真实项目（bishengir / Triton）。

```
笔记路径: docs/mlir/L05 ~ L07
```

| 步骤 | 笔记 | 对标项目 | 产出 |
|------|------|---------|------|
| 3.1 | L05 Toy Mini 手写 | 对照 Toy Tutorial Ch1-2 | 三项目对照表 |
| 3.2 | L06 Triton MLIR 体系 | triton-ascend 源码 | TT / TritonGPU Dialect 分析 |
| 3.3 | L07 triton-ascend 后端 | ascend_interpreter.py | Python ↔ C++ 对接层 |
| 3.4 | L08 bishengir-demo | 三个用例全跑通 | 可运行验证 |

**关键突破**: 理解 Triton + bishengir 如何组成完整的 Ascend 编译链路。

---

## 四、工程项目详情

### 4.1 ⭐ bishengir-demo — 可运行降级流水线

用标准 `mlir-opt` 模拟 bishengir 三阶段降级过程。

#### bishengir 对应

| 阶段 | bishengir (实际) | 本 demo (标准 MLIR) | 共同概念 |
|------|------------------|--------------------|---------|
| 输入 | `linalg.generic` | `linalg.generic` | Linalg dialect |
| Pass1 | `-convert-linalg-to-hfusion` | `--convert-linalg-to-affine-loops` | 高级→中级 IR |
| Pass2 | `-convert-arith-to-hfusion` | `--lower-affine` | 算术操作处理 |
| Pass3 | `-convert-hfusion-to-hivm` | `--convert-scf-to-cf --convert-func-to-llvm` | 最终 IR |
| 输出 | `hivm.load/vadd/store` | `llvm.load/add/store` | 目标相关指令 |

#### 测试结果

```
向量加法 (vecadd_128.mlir):
  Linalg: 3 行  →  Affine: 18 行  →  LLVM: 38 行  (12.7×)
  ✅ mlir-opt --convert-linalg-to-affine-loops 通过
  ✅ 完整降级到 LLVM IR 通过

矩阵乘法 (matmul_4x4x4.mlir):
  Linalg: 1 行  →  Affine: 18 行  →  LLVM: 74 行  (74×)
  ✅ 基础降级通过
  ⚠️ 三重循环完全展开，提供 4 种优化方案对比

融合操作 (fused_128.mlir):
  Linalg: 15 行  →  Affine: 20 行  →  LLVM: 59 行  (3.9×)
  ✅ add + mul 连续操作，展示融合理念
```

#### matmul 的 74× 膨胀与优化

| Variant | 策略 | LLVM 行数 | vs 基准 | 对应 bishengir |
|---------|------|-----------|---------|---------------|
| **V0** | 无优化 (基准) | 74 行 | - | — |
| **V1** | 循环分块 (tile=2x2x1) | 76 行 | +2 行 | — |
| **V2** | 向量化 (tile+vectorize) | 77 行 | +3 行 | `-convert-hfusion-to-hivm` 生成向量指令 |
| **V3** | **硬件映射 (模拟 mmul)** | **5 行** | **-69 行 (-93%)** | `hfusion.cube_matmul → hivm.mmul` |

V3 的 5 行 vs 74 行的差距，正是 bishengir 实际采用的方案——**保持高级语义不展开，直接映射到硬件 Cube 单元**。

```bash
# 运行对比
cd projects/bishengir-demo && bash variants/compare.sh
```

### 4.2 toy-mini — 从零写 Toy 语言解析器

| 特性 | 值 |
|------|-----|
| 语言 | C++17（纯标准库，零外部依赖）|
| 行数 | 1,412 行 |
| 编译 | `g++ -std=c++17 -o toymini toymini.cpp` |
| 输入 | 含 Lexer(14 tokens) + Parser(递归下降) + AST(8 节点) + MLIR Gen |

**支持语法示例**:
```toy
def matmul(M, N, K, A, B, C) {
  var sum = 0.0;
  for k = 0..K {
    sum = sum + A[M][k] * B[k][N];
  }
  C[M][N] = sum;
  return;
}
```

### 4.3 standalone-mlir — 从零构建 MLIR dialect

**6 个 Op**: constant / add / mul / transpose / print / return
**2 个 Pass**: `-count-ops` (分析) + `-elim-transpose` (转换)
**2 种构建**: CMake + Makefile 双方案
**1 个入口**: `standalone-opt` (类似 `bishengir-opt`)

### 4.4 bishengir-op-counter — 自定义 Pass 参考代码

| 文件 | 类型 | 模式 | 对应 Toy Tutorial |
|------|------|------|-------------------|
| `BishengirOpCounter.cpp` | 分析 Pass | `op->walk()` | Ch3 ShapeInferencePass |
| `BishengirPeelTranspose.cpp` | 转换 Pass | `OpRewritePattern` | Ch3 ToyCombine |

---

## 五、快速入门

### 5.1 环境准备

```bash
# 检查环境
bash setup.sh

# 如果缺依赖
brew install llvm cmake
xcode-select --install
```

### 5.2 跑 bishengir-demo（5 分钟）

```bash
cd projects/bishengir-demo
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# 单个用例
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir

# 完整降级到 LLVM IR
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  test-cases/vecadd_128.mlir

# 批量运行
bash run-demo.sh
```

**预期输出**: 每行 `%sum = arith.addf ...` 在 affine 阶段变成 `%sum = affine.load + arith.addf + affine.store`；到 LLVM 阶段变成 `%val = llvm.load + %sum = llvm.add + llvm.store`。

### 5.3 跑 Toy Mini（3 分钟）

```bash
cd projects/toy-mini
g++ -std=c++17 -o toymini toymini.cpp
./toymini
```

**预期输出**: AST 树状结构和 MLIR 风格 IR 文本。

### 5.4 编译 standalone-mlir（10 分钟）

```bash
cd projects/standalone-mlir
export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 查看 MLIR 输入
./build/standalone-opt test/example.mlir

# 跑自定义 Pass
./build/standalone-opt test/example.mlir --count-ops
```

### 5.5 快速调试技巧

```bash
# 查看 IR 的某个 stage
mlir-opt --print-ir-after=<pass-name> input.mlir

# 只跑某个 pass
mlir-opt --pass-pipeline="builtin.module(func.func(count-ops))" input.mlir

# 查看可用 pass
mlir-opt --help | grep standalone
```

---

## 六、三项目技术对照

| 维度 | LLVM IR | MLIR | bishengir |
|------|---------|------|-----------|
| **设计哲学** | 单一 IR | 多层 IR (dialect) | 专用 dialect 链 |
| **类型系统** | `iN`, `ptr`, `struct` | `tensor<T>`, `memref<T>` | `hfusion.tensor<T>` |
| **操作** | 指令 (add/load/store) | Operation (可嵌套) | hivm.vadd/madd |
| **优化** | Pass (FunctionPass) | Pass + Pattern Rewriting | ConversionTarget |
| **降级** | 前端 → IR → 后端 | dialect → dialect → ... | Linalg → HFusion → HIVM |
| **元编程** | TableGen (指令描述) | TableGen (dialect 定义) | TableGen |

### 对应关系

```text
Triton Python kernel
  ↓ Frontend
Triton IR (tt.load/tt.dot/tt.store)
  ↓ [本项目的分析对象]
ascendnpu-ir (bishengir)
  ├── LinalgToHFusion   →  Linalg ops  →  HFusion ops
  ├── ArithToHFusion    →  Arith ops    →  HFusion ops
  └── HFusionToHIVM     →  HFusion ops  →  HIVM ops (NPU)
      ↓
CANN Runtime (华为 SDK, 硬件执行)
```

---

## 七、依赖与环境

| 工具 | 版本 | 安装方式 | 用途 |
|------|------|---------|------|
| LLVM/MLIR | 22.1.6 | `brew install llvm` | mlir-opt, mlir-tblgen |
| cmake | ≥ 3.20 | `brew install cmake` | standalone-mlir 构建 |
| ninja | (可选) | `brew install ninja` | 加速构建 |
| g++/clang++ | C++17 | `xcode-select --install` | Toy Mini 编译 |
| python3 | ≥ 3.8 | 系统自带 | 用例生成器 |

**环境变量**:

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
```

---

## 八、笔记与项目对照表

| 笔记 | 对应项目 | 核心知识点 |
|------|---------|-----------|
| `MLIR-L00` | bishengir-demo | bishengir 三段降级全景 |
| `MLIR-L01` | toy-mini | TableGen dialect 定义 |
| `MLIR-L02` | bishengir-op-counter | Pattern Rewriting 模式 |
| `MLIR-L03` | bishengir-op-counter | Pass 架构：分析 vs 转换 |
| `MLIR-L04` | standalone-mlir | CMake + Makefile + LLVM22 适配 |
| `MLIR-L05` | toy-mini | 解析器四组件架构 |
| `MLIR-L06` | — | Triton TT/TritonGPU 双 Dialect |
| `MLIR-L07` | — | triton-ascend 后端对接 |
| `MLIR-L08` | bishengir-demo | 3 用例 mlir-opt 验证 |

---

## 九、License

MIT License. 详见 [LICENSE](LICENSE)。

---

## 十、相关资源

- [LLVM 官方文档](https://llvm.org/docs/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [Triton 官网](https://triton-lang.org/)
- [bishengir (ascendnpu-ir)](https://github.com/nousresearch/ascendnpu-ir) — 注：需内部访问
- [Triton 源码 (triton-ascend)](https://github.com/openai/triton) — Ascend 分支
