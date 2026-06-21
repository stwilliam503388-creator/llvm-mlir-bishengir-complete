# 004：从 Triton 到 Ascend（贯穿全项目的完整路径）

> 阅读时间：5 分钟 | 前置知识：Primer 00, 01, 02

---

## 4.1 最终目标是什么？

写一段 Triton Python 代码，让它跑到 Ascend NPU 上：

```python
@triton.jit
def vecadd_kernel(A, B, C, N):
    pid = tl.program_id(0)
    offset = pid * 128
    a = tl.load(A + offset)
    b = tl.load(B + offset)
    c = a + b
    tl.store(C + offset, c)
```

从这 8 行 Python，到 NPU 芯片上实际执行的电路信号，中间经过的每一步，**就是本项目的内容**。

---

## 4.2 完整路径图

```
                    Triton Python 代码
                          │
    ┌─────────────────────┴──────────────────────┐
    │                  AST 构建                    │
    │  你的 Triton 代码 → Python AST → Triton AST  │
    └─────────────────────┬──────────────────────┘
                          │   概念: AST (Primer 01)
                          ▼
    ┌─────────────────────┬──────────────────────┐
    │                  IR 生成                    │
    │  Triton AST → Triton IR (tt dialect)        │
    │  tt.load, tt.dot, tt.store 等操作           │
    └─────────────────────┬──────────────────────┘
                          │   概念: IR (Primer 01)
                          ▼
    ┌─────────────────────┬──────────────────────┐
    │              Pass + Lowering               │
    │  Triton IR → TritonGPU IR → LLVM IR        │
    │  加内存布局        降级到 CPU 指令          │
    └──────────┬──────────┴──────────┬───────────┘
               │                     │
               │  概念: Pass, Lowering (Primer 02)
               │  概念: Dialect (Primer 02)
               ▼                     ▼
    ┌────────────────────┐  ┌────────────────────┐
    │    AscendNPU-IR 路径   │  │   标准 LLVM 路径    │
    │  (本项目重点)       │  │  (CPU/GPU)         │
    │                    │  │                    │
    │  TT IR → AIR       │  │  TT IR → LLVM IR   │
    │           ↓        │  │         ↓          │
    │  AIR → HIVM        │  │  LLVM IR → 机器码  │
    │           ↓        │  │         ↓          │
    │  HIVM → CANN       │  │  在 CPU/GPU 执行    │
    │           ↓        │  │                    │
    │  Ascend NPU 执行   │  │                    │
    └────────────────────┘  └────────────────────┘
```

---

## 4.3 本项目覆盖了哪些路段

```
本项目覆盖的范围（用 ████ 标记）：

  Triton Python ──→ AST ──→ Triton IR ──→ AscendNPU-IR ──→ Ascend NPU
                              ██████████████████████████████████
                              ↑                              ↑
                          docs/llvm/                     bishengir-demo
                          docs/mlir/                     bishengir-op-counter
                          standalone-mlir                docs/mlir/L06-L07
                          toy-mini

  你没覆盖到的（但也不需要硬件）：
    ──→ Triton Python AST 构建           (triton-ascend 内部)
    ──→ CANN 运行时对接                    (需要 NPU 硬件)
```

对应关系表：

| 项目/文档 | 覆盖的概念 | 学到什么 |
|-----------|-----------|---------|
| `docs/primer/00-03` | 编译器基础概念 | 用 AI 工程师能懂的语言理解 AST/IR/Pass/Lowering |
| `docs/llvm/` | LLVM IR、SSA、指令、Pass | 读懂 `.ll` 文件，理解 IR 的细节 |
| `docs/mlir/L00-L02` | MLIR dialect、Toy Tutorial | 理解 MLIR 的多层 IR 哲学 |
| `docs/mlir/L03-L04` | 自定义 Pass、Standalone 构建 | 从读到写，产出自己的 dialect |
| `projects/toy-mini` | AST 构建 + IR 生成 | 手写一个语言前端 |
| `projects/standalone-mlir` | dialect 定义 + 构建 | CMake + TableGen |
| `projects/bishengir-demo` | Lowering 全过程 | 实际跑通 3 个用例 |
| `projects/bishengir-op-counter` | 分析 + 转换 Pass | 手写自定义 Pass |
| `docs/mlir/L05-L07` | Triton MLIR 体系 | 理解 triton-ascend 全貌 |

---

## 4.4 读完 Primer 后你应该做什么

```
Step 1: 读 docs/llvm/L00-速通总览.md
        了解 LLVM IR 的全貌（10 分钟）

Step 2: 运行 bishengir-demo
        cd projects/bishengir-demo
        bash variants/compare.sh
        感受从 1 行到 74 行的 Lowering 过程（5 分钟）

Step 3: 回到 LLVM 笔记
        遇到不理解的概念，回到 Primer 找解释
        或者问自己：这个概念属于 AST/IR/Pass/Lowering 中的哪一个？

Step 4: 动手实践
        改一个 .mlir 文件，改参数重新运行
        修改 toymini.cpp 的测试代码
        这是理解最快的方式
```

---

## 4.5 快速自测

1. **从 Triton Python 到 Ascend NPU，中间经过了几次 IR 转换？**
   - 答：至少 4 次（Triton AST → tt dialect → TritonGPU → AIR → HIVM → NPU），取决于具体路径

2. **本项目中哪个工程最接近"真正的编译器降级"？为什么？**
   - 答：bishengir-demo。因为它用 mlir-opt 实际跑通了从 Linalg 到 LLVM 的完整降级过程

3. **学完本项目后，你能做什么？**
   - 读懂 `bishengir-opt` 的 Pass 管线输出
   - 理解 Triton 报错中的 MLIR IR 信息
   - 为 AscendNPU-IR 编写自定义转换 Pass
   - 理解 ascendnpu-ir 源码中每个 Conversion Pass 的作用
