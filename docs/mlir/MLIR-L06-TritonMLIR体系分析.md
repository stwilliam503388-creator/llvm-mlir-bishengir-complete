---
created: 2026-06-21
tags: [triton, mlir, ascend, dialect]
aliases: [Triton MLIR 体系, Triton 编译栈]
---

# Triton MLIR 体系分析

> 基于 `triton-ascend` 源码分析 Triton 如何用 MLIR 编译到 Ascend NPU。
> 对照 bishengir 和 Toy Tutorial，完整理解多级 IR 编译器栈。

---

## 一、整体架构

```
Triton Python 源码 (kernel 函数)
       │
       ▼
┌──────────────────────────────┐
│  Python AST → Triton IR     │  ← Python 端
│  (triton/language/  )       │
└──────────┬───────────────────┘
           │ Triton IR (TIR)
           ▼
┌──────────────────────────────┐
│  TT Dialect (Triton)         │  ← MLIR Dialect 层
│  ● addptr, load, store       │
│  ● dot, matmul               │  include/triton/Dialect/Triton/
│  ● broadcast, reshape        │  TritonOps.td (~1400 行)
└──────────┬───────────────────┘
           │ TT → TritonGPU
           ▼
┌──────────────────────────────┐
│  TritonGPU Dialect           │  ← GPU 特定优化
│  ● memory layouts            │  include/triton/Dialect/TritonGPU/
│  ● shared memory             │  TritonGPUOps.td
│  ● warp specialization       │
└──────────┬───────────────────┘
           │ ConvertTritonToTritonGPU
           ▼
┌──────────────────────────────┐
│  Conversion Passes           │  ← 降级到目标硬件
│  ● ConvertTritonToTritonGPU  │  lib/Conversion/
│  ● ConvertTritonGPUToLLVM    │
│  ● (ConvertToAIR)            │
└──────────┬───────────────────┘
           │ LLVM IR / AIR
           ▼
        CPU / GPU / Ascend NPU
```

---

## 二、核心 Dialect

### 2.1 TT Dialect（Triton IR）

**位置**：`include/triton/Dialect/Triton/IR/TritonOps.td`（1416 行）

基类：
```tablegen
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
}
```

**关键 op**：

| Op | 功能 | 对应 Toy Tutorial |
|----|------|------------------|
| `tt.addptr` | 指针加法（偏移） | — |
| `tt.load` | 从内存加载数据 | — |
| `tt.store` | 写回内存 | — |
| `tt.dot` | 矩阵乘法（核心） | — |
| `tt.broadcast` | 广播到指定形状 | — |
| `tt.trans` | 转置 | `toy.transpose` |
| `tt.reduce` | 归约（sum/max） | — |
| `tt.reshape` | 改变张量形状 | — |
| `tt.call` | 函数调用 | — |
| `tt.return` | 函数返回 | `toy.return` |

### 2.2 TritonGPU Dialect

**位置**：`include/triton/Dialect/TritonGPU/`

```
TritonGPU 在 TT 的基础上增加了 GPU 硬件相关的信息：
  ● Memory layout（共享内存布局）
  ● Warp 级别调度
  ● 分布式内存访问模式
```

**核心概念对比**：

| 概念 | TT Dialect | TritonGPU Dialect |
|------|-----------|------------------|
| **张量** | 逻辑张量 | **物理布局**的张量 |
| **编码** | 无 | `BlockedEncoding`, `MmaEncoding` |
| **优化** | 通用 | **Warp 专用化、流水线** |
| **降级目标** | - | LLVM / AIR / PTX |

---

## 三、Triton → Ascend 路径

### 当前实现

```
Triton 源码
    │
    ▼
┌───────────────────────┐
│  triton-ascend        │  ← Python 层对接
│  (python/backend/     │
│   ascend/)            │
└───────┬───────────────┘
        │ TIR (Triton IR)
        ▼
┌───────────────────────┐
│  ascendnpu-ir         │  ← MLIR 层
│  (bishengir)          │
│  ● TIR → AIR          │
│  ● AIR → HIVM         │
└───────┬───────────────┘
        │ HIVM IR
        ▼
┌───────────────────────┐
│  CANN (昇腾 SDK)      │  ← 硬件后端
│  ● 毕昇编译器         │
│  ● NPU 驱动           │
└───────────────────────┘
```

### triton-ascend 仓库的 MLIR 相关文件

```
triton-ascend/
├── include/triton/
│   ├── Dialect/
│   │   ├── Triton/IR/           ← TT dialect 定义
│   │   │   ├── TritonOps.td     (1416行，op 定义)
│   │   │   ├── TritonDialect.td (dialect 声明)
│   │   │   ├── TritonTypes.td   (类型定义)
│   │   │   └── TritonAttrDefs.td (属性定义)
│   │   └── TritonGPU/IR/        ← TritonGPU dialect
│   │       └── TritonGPUOps.td  (GPU 相关 op)
│   └── Conversion/              ← 转换 Pass 头文件
│       ├── TritonToTritonGPU/
│       └── TritonGPUToLLVM/
├── lib/
│   ├── Dialect/                 ← dialect 实现
│   ├── Conversion/              ← 转换 Pass 实现
│   └── Target/                  ← 目标硬件描述
└── python/triton/               ← Python 前端
```

---

## 四、三项目对照总结

| 层面 | Toy Tutorial | bishengir (ascendnpu-ir) | Triton (triton-ascend) |
|------|-------------|------------------------|----------------------|
| **编程模型** | Toy 语言 | MLIR (Linalg dialect) | **Triton Python kernel** |
| **高级 IR** | `toy.constant/add/mul` | `linalg.generic` | `tt.load/dot/store` |
| **中间 IR** | `affine.for + arith` | `hfusion.elemwise_binary` | `TritonGPU (layout)` |
| **低级 IR** | `scf + memref` | `hivm.vadd/load/store` | LLVM IR / AIR |
| **最终目标** | CPU（LLVM JIT） | **Ascend NPU** | CPU / GPU / **Ascend** |
| **Dialect 数** | 1 (toy) | 8 (hfusion, hivm 等) | 2 (TT + TritonGPU) |
| **Pass 注册方式** | `PassRegistration` | `InitAllPasses.h` | `registerTritonPasses()` |

### Triton 与 bishengir 的 MLIR 使用对比

```
Triton:
  Triton Python → tt dialect → TritonGPU → LLVM/AIR
                     ↑ 通用MLIR      ↑ GPU优化     ↑ 后端

bishengir:
  MLIR (linalg) → hfusion → hivm → CANN
                     ↑ 算子融合   ↑ NPU指令    ↑ 华为SDK
```

**关键差异**：
- Triton 先保证**通用性**（CPU/GPU 都能跑），再通过 TritonGPU 做硬件优化
- bishengir 直接针对**昇腾 NPU** 设计，dialect 更贴近硬件
- 两者共享相同的 MLIR 基础设施（TableGen、Pattern Rewriting、Dialect Conversion）

---

## 五、Triton 的 MLIR Pass 流水线

Triton 的编译管线由多个 Pass 组成（在 `python/triton/compiler.py` 中编排）：

```python
# 简化自 triton/compiler.py
def compile(kernel_fn):
    # 1. 生成 Triton IR (Python AST → TT dialect)
    ir = frontend(kernel_fn)

    # 2. 应用转换 Pass
    pm = PassManager(ir)
    pm.add_pass(ConvertTritonToTritonGPU())  # 添加内存布局
    pm.add_pass(TritonGPUPipeline())          # warp 流水线
    pm.add_pass(ConvertTritonGPUToLLVM())     # 降级到 LLVM IR

    # 3. 后端编译
    return backend.compile(ir, target="ascend")
```

**对应 bishengir 的 Pass 流水线**：

```bash
# bishengir 等价操作
bishengir-opt \
  -convert-linalg-to-hfusion \    # ≅ ConvertTritonToTritonGPU
  -convert-arith-to-hfusion \
  -convert-hfusion-to-hivm \      # ≅ ConvertTritonGPUToLLVM
  vecadd.mlir
```

---

## 六、关键文件路径

```
# Triton 核心 MLIR
triton-ascend/include/triton/Dialect/Triton/IR/TritonOps.td
triton-ascend/lib/Dialect/Triton/IR/

# TritonGPU
triton-ascend/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td
triton-ascend/lib/Dialect/TritonGPU/IR/

# 转换 Pass
triton-ascend/lib/Conversion/TritonToTritonGPU/
triton-ascend/lib/Conversion/TritonGPUToLLVM/

# bishengir (ascendnpu-ir)
ascendnpu-ir/bishengir/include/bishengir/Dialect/
ascendnpu-ir/bishengir/lib/Conversion/

# Toy Tutorial
toy-tutorial/src/Ch2~Ch6/
```
