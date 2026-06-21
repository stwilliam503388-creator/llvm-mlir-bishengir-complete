# 完整输出总结文档

> 生成时间: 2026-06-21
> 模型: deepseek-v4-flash
> 环境: macOS 26.5.1, Apple Silicon, LLVM 22.1.6 (Homebrew)

---

## 一、任务总览

本次连续多轮对话覆盖 LLVM → MLIR → bishengir (Ascend NPU) 全链路学习，最终交付可运行的降级 Demo。

### 执行阶段

| 阶段 | 内容 | 产出 |
|------|------|------|
| **0. LLVM 基础** | llvm-ir-tutorial 中文教程速通 | 7 篇 Obsidian 笔记 |
| **1. MLIR 概念** | dialect/operation/pass/pattern 体系 | 2 篇基础笔记 + mlir-opt 实操 |
| **2. Toy Tutorial** | 官方 Toy 教程 7 章源码逐行解读 | 2 篇笔记 + 三项目对照表 |
| **3. 自定义 Pass** | 针对 bishengir 写分析/转换 Pass | 2 个 C++ 源码文件 |
| **4. Standalone 项目** | 从零构建 CMake + Makefile 项目 | CMakeLists + .td + 入口文件 |
| **5. Toy Mini 手写** | 纯 C++17 零依赖解析器 | 1412 行，编译通过 |
| **6. Triton 体系** | triton-ascend 源码分析 | 2 篇笔记 + 全链路对照 |
| **7. ⭐ bishengir-demo** | 可运行 mlir-opt 降级流水线 | 3 用例 + 生成器 + 运行脚本 |

---

## 二、知识库产出（15 篇笔记）

### LLVM 基础（7 篇）

| 文件 | 核心内容 |
|------|---------|
| **L00-速通总览** | 学习路线图 + 资源链接 + 三项目对应关系 |
| **L01-架构与HelloWorld** | 三段式架构（前端→IR→后端）、SSA、.ll 文件格式 |
| **L02-类型系统与GEP** | iN 类型、类型转换、GetElementPtr 剥洋葱模型 |
| **L03-控制流与Phi节点** | br 指令、φ 节点汇合、for 循环 φ 自引用 |
| **L04-内置函数与属性** | llvm.memcpy、expect 内建、属性组 |
| **L05-Pass开发** | New PM 架构、FunctionPass 骨架、BBCounter 实战 |
| **L06-IR速查表** | 常用指令 + 调试命令 + FileCheck |

### MLIR 体系（8 篇，本期核心）

| 文件 | 核心内容 | 工程关联 |
|------|---------|---------|
| **L00-速通与bishengir实战** | dialect/operation/region 概念 + 三阶段降级对照 | bishengir-demo |
| **L01-ToyTutorial Ch1-2** | Toy 语法 + TableGen + MLIRGen | toy-mini |
| **L02-ToyTutorial Ch3-6** | Pattern Rewriting + ConversionTarget | bishengir-op-counter |
| **L03-自定义Pass实战** | OpCounter (walk) + PeelTranspose (OpRewritePattern) | bishengir-op-counter |
| **L04-Standalone实战** | cmake 4.3 兼容性 + Makefile 方案 + LLVM 22 适配 | standalone-mlir |
| **L05-ToyMini手写** | Lexer/Parser/AST/IR 四组件全实现 | toy-mini |
| **L06-TritonMLIR体系** | TT dialect (1416 ops) + TritonGPU + 三项目对照 | — |
| **L07-triton-ascend后端** | ascend_interpreter.py + CANN + Python/C++ 对接层 | — |
| **L08-bishengir-demo** | 3 用例全部 mlir-opt 验证通过 | bishengir-demo |

---

## 三、工程交付（4 个项目）

### 1. ⭐ bishengir-demo — 可运行降级流水线

**位置**: `projects/bishengir-demo/`

#### 验证结果

| 用例 | 输入 (Linalg) | Stage1 (affine) | Stage3 (LLVM) | 膨胀率 | 说明 |
|------|-------------|----------------|--------------|--------|------|
| vecadd_128 | 3 行 | 18 行 | 38 行 | 12.7× | 最简向量加法 |
| matmul_4x4x4 | 1 行 | 18 行 | 72 行 | **72×** | 三重循环展开 |
| fused_128 | 15 行 | 20 行 | 59 行 | 3.9× | add + mul 连续 |

#### bishengir 对应

```
bishengir:        Linalg → HFusion.elemwise_binary → HIVM.load/vadd/store
标准 MLIR (本):   Linalg → affine.for + arith.addf → llvm.load/add/store
```

#### 文件清单

```
bishengir-demo/
├── README.md                    — 使用说明
├── bishengir-demo.py            — Python 生成器（可扩展用例）
├── run-demo.sh                  — 批量运行脚本
└── test-cases/
    ├── vecadd_128.mlir           — 向量加法 ✅
    ├── matmul_4x4x4.mlir         — 矩阵乘法 ✅
    └── fused_128.mlir            — 融合操作 ✅
```

#### 运行方式

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir
mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm test-cases/vecadd_128.mlir
```

---

### 2. toy-mini — 从零写 Toy 解析器

**位置**: `projects/toy-mini/toymini.cpp`

| 特性 | 详情 |
|------|------|
| **语言** | C++17（纯标准库，零依赖）|
| **行数** | 1,412 行 |
| **编译** | `g++ -std=c++17 -o toymini toymini.cpp` ✅ |
| **Lexer** | 14 token 类型（关键字/运算符/数字/标识符）|
| **Parser** | 递归下降（支持优先级控制）|
| **AST** | 8 种节点类型（Number/Var/Binary/Call/Literal/VarDecl/Print/Return）|
| **IR 生成** | MLIR 风格文本输出（对照 Toy Tutorial Ch2）|

**支持语法**:
```
def name(params) { var x = [[1,2],[3,4]]; print(x + y); return z; }
```

---

### 3. standalone-mlir — 从零构建 MLIR dialect

**位置**: `projects/standalone-mlir/`

| 组件 | 详情 |
|------|------|
| **Dialect** | `standalone`（6 ops: constant/add/mul/transpose/print/return）|
| **TableGen** | `StandaloneOps.td`（mlir-tblgen 语法验证通过）|
| **构建** | CMake + Makefile 双方案 |
| **入口** | `standalone-opt.cpp`（含 2 个 Pass）|
| **Pass** | `count-ops`（walk 分析）+ `elim-transpose`（Pattern 转换）|
| **MLIR 依赖** | 手动配置（`/opt/homebrew/opt/llvm`，跳过了 `find_package` 的 AddMLIR 冲突）|

**编译状态**: mlir-tblgen 生成 .inc 成功 ✅，完整编译需要 LLVM 22 的 Properties 机制支持（已记录适配方案）

---

### 4. bishengir-op-counter — 自定义 Pass 源码

**位置**: `projects/bishengir-op-counter/`

| 文件 | 类型 | 对应 Toy Tutorial | 功能 |
|------|------|-------------------|------|
| `BishengirOpCounter.cpp` | 分析 Pass | Ch3 ShapeInferencePass (walk) | 统计 hfusion/hivm ops 分布 |
| `BishengirPeelTranspose.cpp` | 转换 Pass | Ch3 ToyCombine + Ch5 Lowering | 消除冗余 transpose，融合 add→mul |

---

## 四、关键技术对照

### bishengir 三阶段 vs 本 Demo

| 阶段 | bishengir | 本 Demo (标准 MLIR) | 共同概念 |
|------|-----------|--------------------|---------|
| 1 | `-convert-linalg-to-hfusion` | `--convert-linalg-to-affine-loops` | `linalg.generic` → 更低级 IR |
| 2 | `-convert-arith-to-hfusion` | `--lower-affine` | 处理算术操作 |
| 3 | `-convert-hfusion-to-hivm` | `--convert-scf-to-cf --convert-func-to-llvm` | 最终降级到目标 IR |

### 三项目对照

| 维度 | Toy Tutorial | bishengir (ascendnpu-ir) | Triton (triton-ascend) |
|------|-------------|------------------------|----------------------|
| **编程模型** | Toy 语言 | MLIR (Linalg dialect) | **Triton Python kernel** |
| **高级 IR** | `toy.constant/add/mul` | `linalg.generic` | `tt.load/dot/store` |
| **中间 IR** | `affine.for + arith` | `hfusion.elemwise_binary` | `TritonGPU (layout)` |
| **低级 IR** | `scf + memref` | `hivm.vadd/load/store` | LLVM IR / AIR |
| **最终目标** | CPU (LLVM JIT) | **Ascend NPU** | CPU / GPU / **Ascend** |
| **Dialect 数** | 1 (toy) | 8 (hfusion, hivm 等) | 2 (TT + TritonGPU) |

---

## 五、运行环境

### 依赖

| 工具 | 版本 | 安装方式 |
|------|------|---------|
| LLVM/MLIR | 22.1.6 | `brew install llvm` |
| cmake | 4.3.2 | `brew install cmake` |
| ninja | 1.13.2 | `brew install ninja` |
| g++ / clang++ | (Xcode CLT) | `xcode-select --install` |

### 环境变量

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
# 或
export LLVM_DIR="/opt/homebrew/opt/llvm"
export PATH="$LLVM_DIR/bin:$PATH"
```

---

## 六、附录：文件清单

### 文档 (15 篇)

```
docs/llvm/ (7 篇, ~45KB)
  LLVM-L00.md .. LLVM-L06.md

docs/mlir/ (8 篇, ~76KB)
  MLIR-L00.md .. MLIR-L08.md
```

### 代码 (13 个文件)

```
projects/
├── bishengir-demo/       (6 files)
├── toy-mini/             (1 file)
├── standalone-mlir/      (7 files)
└── bishengir-op-counter/ (2 files)
```

### 配置 (3 个)

```
projects/standalone-mlir/CMakeLists.txt
projects/standalone-mlir/Makefile
projects/bishengir-demo/run-demo.sh
```
