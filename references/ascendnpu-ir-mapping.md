# AscendNPU-IR 源码对接追踪

> 本文件将本项目的概念/代码/文档与 AscendNPU-IR 官方源码和文档建立对应关系。
> 读者拿到 AscendNPU-IR 源码后，可根据此文件快速定位。

**参考链接**:
- 官方代码仓: https://github.com/Ascend/AscendNPU-IR
- 中文文档: https://ascendnpu-ir.gitcode.com/zh_cn/index.html
- 本项目分析的 fork: `~/hermes-workspace/ascendnpu-ir/`（即 AscendNPU-IR 核心编译器代码，源码目录名为 `bishengir/`）

---

## 1. 三阶段降级源码追踪

AscendNPU-IR 的核心是三阶段降级：Linalg → HFusion → HIVM。
每个阶段在 ascendnpu-ir 源码中对应一个 Conversion Pass 目录。

### 1.1 Pass1: Linalg → HFusion

| 项 | 内容 |
|----|------|
| **目录** | `bishengir/lib/Conversion/LinalgToHFusion/` |
| **主文件** | `LinalgToHFusion.cpp` |
| **核心类** | `ConvertLinalgToHFusion` (Pass 入口) |
| **Pattern 模式** | `OpRewritePattern<linalg::GenericOp>` 匹配 linalg 操作 |
| **输出** | `hfusion.elemwise_binary` / `hfusion.cube_matmul` |
| **匹配条件** | `linalg::GenericOp` 的 body 中只含 `arith.addf` 等基本运算 |
| **测试用例** | `test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir` |

**关键函数**:

```cpp
// LinalgToHFusion.cpp 中的核心 Pattern
class LinalgGenericToHFusion : public OpRewritePattern<linalg::GenericOp> {
    LogicalResult matchAndRewrite(linalg::GenericOp op, ...) override {
        // 1. 验证: linalg.generic 是否满足转换条件
        // 2. 根据 body 中的运算选择 hfusion op 类型
        //    arith.addf → hfusion.elemwise_binary {fun = add}
        //    arith.mulf → hfusion.elemwise_binary {fun = mul}
        // 3. 生成 hfusion op，替换原 linalg.generic
    }
};
```

### 1.2 Pass2: Arith → HFusion

| 项 | 内容 |
|----|------|
| **目录** | `bishengir/lib/Conversion/ArithToHFusion/` |
| **主文件** | `ArithToHFusion.cpp` |
| **核心类** | `ConvertArithToHFusion` |
| **处理 op** | `arith.addf`, `arith.mulf`, `arith.cmpf` 等 |
| **测试用例** | `test/Conversion/ArithToHFusion/arith-to-hfusion.mlir` |

### 1.3 Pass3: HFusion → HIVM

| 项 | 内容 |
|----|------|
| **目录** | `bishengir/lib/Conversion/HFusionToHIVM/` |
| **主文件** | `HFusionToHIVM.cpp` |
| **核心类** | `ConvertHFusionToHIVM` |
| **输出** | `hivm.load` / `hivm.vadd` / `hivm.mmul` / `hivm.store` |
| **测试用例** | `test/Conversion/HFusionToHIVM/hfusion-to-hivm.mlir` |

**关键区别**:
- HFusion 是**算子级 IR**（`hfusion.elemwise_binary` 表示"执行逐元素运算"）
- HIVM 是**指令级 IR**（`hivm.vadd` 表示"执行向量加法指令"）
- 类比: HFusion = "做一盘炒鸡蛋"，HIVM = "打蛋→热油→炒→装盘"

---

## 2. Dialect 源码追踪

### 2.1 HFusion Dialect

| 项 | 内容 |
|----|------|
| **定义文件** | `bishengir/include/bishengir/Dialect/HFusion/HFusionOps.td` |
| **实现文件** | `bishengir/lib/Dialect/HFusion/` |
| **核心 Op** | `elemwise_binary`, `elemwise_unary`, `cube_matmul` |
| **TableGen 基类** | `HFusion_Op<string mnemonic>` (类似 Toy Tutorial 的 `Toy_Op`) |

```
        HFusionOps.td
        ├── def ElemwiseBinaryOp
        │   ├── 参数: lhs, rhs, fun (add/mul/sub/div)
        │   ├── 输出: result tensor
        │   └── 约束: lhs 和 rhs 形状相同
        ├── def ElemwiseUnaryOp
        │   ├── 参数: input, fun (exp/sqrt/neg)
        │   └── 输出: result tensor
        └── def CubeMatmulOp
            ├── 参数: A, B, C (memref)
            └── 语义: C += A × B (矩阵乘加)
```

### 2.2 HIVM Dialect

| 项 | 内容 |
|----|------|
| **定义文件** | `bishengir/include/bishengir/Dialect/HIVM/HIVMOps.td` |
| **实现文件** | `bishengir/lib/Dialect/HIVM/` |
| **核心 Op** | `vload`, `vadd`, `vmul`, `vstore`, `mmul` |
| **语义** | 每条指令对应一个 Ascend NPU 硬件指令 |

### 2.3 Annotation / HACC / Scope / Symbol Dialect

| Dialect | 定义文件 | 用途 |
|---------|---------|------|
| **Annotation** | `Dialect/Annotation/AnnotationOps.td` | 标记/注释（用于调试、profiling） |
| **HACC** | `Dialect/HACC/HACCOps.td` | 高级计算控制（循环/同步/调度） |
| **Scope** | `Dialect/Scope/ScopeOps.td` | 作用域管理（内存区域/执行域） |
| **Symbol** | `Dialect/Symbol/SymbolOps.td` | 符号管理（变量名/函数名映射） |

---

## 3. 翻译文档索引

以下 9 篇中文翻译文档在 `docs/ascendnpu-ir/translations/` 目录下。

| # | 文档 | 原始源码路径 | 内容 |
|---|------|------------|------|
| 01 | `AnnotationPasses.md` | `bishengir/docs/cn/Pass/AnnotationPass.md` | Annotation dialect 的转换 Pass |
| 02 | `HACCPasses.md` | `bishengir/docs/cn/Pass/HACCPass.md` | HACC dialect 的转换 Pass |
| 03 | `ScopePasses.md` | `bishengir/docs/cn/Pass/ScopePass.md` | Scope dialect 的转换 Pass |
| 04 | `SymbolPasses.md` | `bishengir/docs/cn/Pass/SymbolPass.md` | Symbol dialect 的转换 Pass |
| 05 | `AnnotationDialect.md` | `bishengir/docs/cn/Dialect/AnnotationDialect.md` | Annotation dialect 定义详解 |
| 06 | `ScopeDialect.md` | `bishengir/docs/cn/Dialect/ScopeDialect.md` | Scope dialect 定义详解 |
| 07 | `SymbolDialect.md` | `bishengir/docs/cn/Dialect/SymbolDialect.md` | Symbol dialect 定义详解 |
| 08 | `MathExtDialect.md` | `bishengir/docs/cn/Dialect/MathExtDialect.md` | 数学扩展 dialect |
| 09 | `MemRefExtDialect.md` | `bishengir/docs/cn/Dialect/MemRefExtDialect.md` | 内存引用扩展 dialect |

---

## 4. 深度分析笔记索引

以下 2 篇深度分析笔记在 `docs/ascendnpu-ir/analysis/` 目录下。

| 文档 | 大小 | 内容概要 |
|------|------|---------|
| `BishengIR代码仓库解读.md` | 21.5KB | 代码仓库结构逐目录解读，含每个子目录的功能说明和关键文件清单 |
| `AscendNPUIR文档总结.md` | 32KB | 官方文档体系的完整总结，含架构图、Pass 管线总览、dialect 关系图 |

---

## 5. 阅读路径

拿到 AscendNPU-IR 源码后，按以下顺序阅读：

```
Step 1: 入口 ── tools/bishengir-opt/bishengir-opt.cpp
  看 Pass 怎么注册、dialect 怎么加载、命令行怎么解析

Step 2: 一个完整 Pass ── lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
  看 ConversionTarget 怎么设、RewritePattern 怎么写、applyPartialConversion 怎么调

Step 3: Dialect 定义 ── include/bishengir/Dialect/HFusion/HFusionOps.td
  看 TableGen 怎么定义 Op、assemblyFormat 怎么写、constraints 怎么加

Step 4: 测试用例 ── test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
  看输入 IR 和输出 IR 的格式，理解 Pass 的效果

Step 5: 自定义 Pass ── 对照本项目的 ascendnpu-ir-op-counter
  按 ascendnpu-ir-op-counter 的注释提示注册到 bishengir-opt
```

**对应本项目的学习路径**:

| 本项目 Stage | 对应 AscendNPU-IR 阅读 |
|-------------|----------------------|
| Stage -1 (Primer) | — |
| Stage 0 (LLVM IR) | — |
| Stage 1 (MLIR 概念) | 读本 mapping 文档 §1 |
| Stage 2 (工程实战) | 读 Step 1→4，跑测试用例 |
| Stage 3 (体系对照) | 读 §2 Dialect 源码 + §3 翻译文档 |

---

## 6. 项目工程源码追踪

### 6.1 ascendnpu-ir-demo ↔ AscendNPU-IR

| 本项目的文件 | 对应 AscendNPU-IR 源码 | 关系 |
|------------|----------------------|------|
| `test-cases/vecadd_128.mlir` | `test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir` | 同类型输入，本 demo 用标准 MLIR |
| `test-cases/matmul_4x4x4.mlir` | `test/Conversion/LinalgToHFusion/matmul-to-hfusion.mlir` | 同类型输入 |
| `test-cases/fused_128.mlir` | 无直接对应 | 演示融合概念 |
| `variants/variant0_baseline.sh` | `bishengir-opt --convert-linalg-to-hfusion` | 等价命令行 |
| `variants/variant3_hw_mapping.sh` | `lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp` | 模式对照 |

### 6.2 ascendnpu-ir-op-counter ↔ AscendNPU-IR

| 本项目的文件 | 对应 AscendNPU-IR 源码 | 模式关系 |
|------------|----------------------|---------|
| `BishengirOpCounter.cpp` | `lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp` | 分析 Pass vs 转换 Pass |
| `BishengirPeelTranspose.cpp` | `lib/Conversion/ToyCombine.cpp` | 同使用 `OpRewritePattern` |

### 6.3 standalone-mlir ↔ AscendNPU-IR

| 本项目的文件 | 对应 AscendNPU-IR 源码 | 模式关系 |
|------------|----------------------|---------|
| `include/standalone/StandaloneOps.td` | `include/bishengir/Dialect/HFusion/HFusionOps.td` | 同 TableGen 语法 |
| `tools/standalone-opt.cpp` | `tools/bishengir-opt/bishengir-opt.cpp` | 同入口模式 |
| `CMakeLists.txt` | `bishengir/CMakeLists.txt` | 同 `find_package(MLIR)` 模式（本 demo 跳过）|

---

## 7. AscendNPU-IR 源码目录速查

| 目录 | 功能 | 在本项目中的对应 |
|------|------|----------------|
| `tools/bishengir-opt/` | 主入口，类似 mlir-opt | `standalone-mlir/tools/standalone-opt.cpp` |
| `include/bishengir/Dialect/` | Dialect 的 .td 定义 | `standalone-mlir/include/standalone/StandaloneOps.td` |
| `lib/Dialect/HFusion/` | HFusion dialect 实现 | — |
| `lib/Dialect/HIVM/` | HIVM dialect 实现 | — |
| `lib/Conversion/LinalgToHFusion/` | Pass1 实现 | ascendnpu-ir-demo Stage1 |
| `lib/Conversion/ArithToHFusion/` | Pass2 实现 | ascendnpu-ir-demo Stage2 |
| `lib/Conversion/HFusionToHIVM/` | Pass3 实现 | ascendnpu-ir-demo Stage3 |
| `test/Conversion/` | 测试用例 | ascendnpu-ir-demo test-cases |
| `docs/cn/` | 中文文档 | `docs/ascendnpu-ir/translations/` |
