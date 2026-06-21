---
created: 2026-06-21
tags: [triton, ascend, backend, compiler]
aliases: [triton-ascend 后端, Ascend 编译管线]
---

# triton-ascend 后端深度分析：Triton → Ascend 全链路

> 基于源码分析 triton-ascend 如何把 Triton Python 代码编译到 Ascend NPU。
> 对接之前学习的 MLIR + bishengir 知识，完成全链路理解。

---

## 一、整体流水线

```
Triton @jit kernel (Python)
       │
       ▼  code_generator.py
┌──────────────────────────────┐
│  Triton IR (MLIR TT dialect) │  ← 中间表示层
│  ● tt.load, tt.dot, etc.     │
└──────────┬───────────────────┘
           │  C++ MLIR Passes
           ▼
┌──────────────────────────────┐
│  MLIR 优化/转换 Pass         │  ← 降级优化层
│  ● TT → TritonGPU            │  lib/Conversion/
│  ● TritonGPU → LLVM          │
└──────────┬───────────────────┘
           │ LLVM IR
           ▼
┌──────────────────────────────┐
│  ascend_interpreter.py       │  ← 运行时解释层
│  ● 接收 LLVM IR / TIR        │
│  ● 映射到 ascendnpu-ir API  │
│  ● 调用 bishengir 转换       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  ascendnpu-ir (bishengir)    │  ← ASCEND 专用层
│  ● Linalg → HFusion → HIVM  │
│  ● CANN 运行时               │
└──────────┬───────────────────┘
           │
           ▼
       Ascend NPU
```

---

## 二、编译管线核心代码

### 2.1 入口：`compiler.py`

位置：`python/triton/compiler/compiler.py`

```python
def compile(fn, signature, **kwargs):
    """Triton 编译入口"""

    # 1. 前端：解析 Python AST → Triton IR (MLIR)
    if not src:
        src = code_generator.generate(fn, signature)

    # 2. MLIR 优化 Pass 管线
    module = parse_mlir(src)
    pm = PassManager(module)

    # 添加标准 Triton 优化 Pass
    pm.add_pass("tt.add_noalias")           # 指针别名分析
    pm.add_pass("tt.vectorize_load_store")  # 向量化加载/存储
    pm.add_pass("tt.coalesce_duplicates")   # 合并重复操作

    # 3. 降级到 LLVM IR
    pm.add_pass("convert_triton_to_llvm")

    # 4. 后端编译（针对目标硬件）
    backend = get_backend(target)
    return backend.compile(module)
```

### 2.2 Ascend 运行时：`ascend_interpreter.py`

位置：`python/triton/runtime/ascend_interpreter.py`（734 行）

```python
class AscendInterpreter:
    """将 Triton IR 解释执行到 Ascend NPU"""

    def __init__(self):
        self.npu_device = get_ascend_device()
        self.cann_api = CANN()

    def launch(self, kernel, grid, args):
        """启动 kernel 执行"""

        # 1. 准备输入数据（Host → NPU 显存）
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                self.npu_device.copy_to_npu(arg)

        # 2. 调用 bishengir / CANN 执行 kernel
        #    这里对接 ascendnpu-ir 的运行时
        self.cann_api.launch_kernel(
            kernel_name=kernel.name,
            grid=grid,
            args=args_ptrs
        )

        # 3. 读取结果（NPU 显存 → Host）
        for name, ptr in outputs:
            self.npu_device.copy_to_host(name, ptr)
```

### 2.3 C++ MLIR Pass 管线

位置：`lib/Conversion/` 和 `lib/Dialect/`

```
Triton C++ MLIR 管线（编译到 LLVM 时）：

  1. tt.addptr → llvm.getelementptr        (地址计算)
  2. tt.load   → llvm.load                  (内存加载)
  3. tt.store  → llvm.store                 (内存存储)
  4. tt.dot    → llvm.matrix_intrinisc      (矩阵乘)

Triton → Ascend 管线（通过 bishengir 时）：

  1. Triton IR → TIR (Triton IR)             ← 前端
  2. TIR → AIR (Ascend NPU IR)               ← ascendnpu-ir
  3. AIR → HIVM 指令                          ← bishengir
  4. HIVM → CANN 可执行代码                    ← 华为 SDK
```

---

## 三、Ascend 专有对接层

### 3.1 bishengir 的作用

```
        Triton IR (tt dialect)
             │
             ▼
    ┌──────────────────────┐
    │    triton-ascend     │  ← Python 端
    │  ascend_interpreter  │     运行时 + 参数管理
    └──────────┬───────────┘
               │ 调用 bishengir API
               ▼
    ┌──────────────────────┐
    │    ascendnpu-ir      │  ← C++ MLIR 层
    │  bishengir-opt       │     转换 Pass 管线
    │  TIR → AIR → HIVM    │
    └──────────┬───────────┘
               │ 通过 CANN 驱动
               ▼
    ┌──────────────────────┐
    │    CANN Runtime      │  ← 华为 NPU SDK
    │  ● 内存管理          │
    │  ● Kernel 启动       │
    │  ● 算子库             │
    └──────────────────────┘
```

### 3.2 关键 Python 工具

| 文件 | 功能 |
|------|------|
| `tools/get_ascend_devices.py` | 检测可用 Ascend 设备 |
| `tools/compile.py` | AOT 编译工具 |
| `runtime/ascend_interpreter.py` | Ascend 运行时解释器 |

---

## 四、Attention Kernel 流水线追踪

`examples/gluon/01-attention-forward.py` 的编译路径：

```python
# 1. 用户代码
@triton.jit
def attention_fwd(Q, K, V, Output, ...):
    # Triton 语言描述注意力计算
    ...

# 2. 编译流程
#    Python AST → Triton IR (TT dialect)
#       │
#       ├── tt.load (加载 Q, K, V)
#       ├── tt.dot  (Q × K^T, 注意力分数)
#       ├── tt.reduce (softmax 归约)
#       ├── tt.dot  (分数 × V)
#       └── tt.store (写回 Output)
#       │
#       ├── [GPU 路径] → TritonGPU → CUDA → GPU
#       └── [Ascend 路径] → TIR → ascendnpu-ir → NPU
```

**Triton MLIR 中的 Attention 关键操作**：

```mlir
// Triton IR 视角（在 ascendnpu-ir 处理前）
module {
  func.func @attention_fwd(%Q: !tt.ptr<f16>,
                           %K: !tt.ptr<f16>,
                           %V: !tt.ptr<f16>,
                           %Output: !tt.ptr<f16>) {
    // 加载 Q, K, V 到寄存器
    %q_reg = tt.load %Q : f16
    %k_reg = tt.load %K : f16
    %v_reg = tt.load %V : f16

    // 矩阵乘法: Q × K^T
    %score = tt.dot %q_reg, %k_reg : f16

    // Softmax（归约）
    %max = tt.reduce %score {op = "max"}
    %exp = tt.elementwise %score, %max {op = "exp"}
    %sum = tt.reduce %exp {op = "sum"}
    %softmax = tt.elementwise %exp, %sum {op = "div"}

    // 矩阵乘法: softmax × V
    %result = tt.dot %softmax, %v_reg : f16

    // 写回
    tt.store %Output, %result
  }
}
```

---

## 五、完整技术对照

| 层面 | Triton | bishengir (ascendnpu-ir) | 共同基础 |
|------|--------|------------------------|---------|
| **Dialect 定义** | `TritonOps.td` (1416行) | `HFusionStructuredOps.td` | **TableGen** |
| **类型系统** | `!tt.ptr`, `!tt.tensor` | `memref`, `tensor` | MLIR 类型框架 |
| **转换 Pass** | `ConvertTritonToTritonGPU` | `ConvertLinalgToHFusion` | **Dialect Conversion** |
| **Pattern 模式** | `OpRewritePattern` | `OpRewritePattern` | 相同 API |
| **Pass 管理** | `PassManager::addPass()` | `PassManager::addPass()` | 相同 API |
| **运行时** | `ascend_interpreter.py` | CANN SDK | Python + C++ 混合 |
| **目标硬件** | GPU / Ascend NPU | **Ascend NPU 专用** | 通用 vs 专用 |

---

## 六、关键文件路径总表

```text
# Triton 端
triton-ascend/python/triton/
├── compiler/compiler.py              ★ 编译入口
├── runtime/ascend_interpreter.py     ★ Ascend 运行时
├── backends/compiler.py              ★ 后端基类
├── backends/driver.py                ★ 驱动接口
├── tools/get_ascend_devices.py        ★ NPU 设备检测
└── tools/compile.py                   ★ AOT 编译

triton-ascend/lib/
├── Dialect/Triton/IR/                ★ TT dialect
├── Dialect/TritonGPU/IR/             ★ TritonGPU dialect
├── Conversion/TritonToTritonGPU/     ★ 转换 Pass
├── Conversion/TritonGPUToLLVM/       ★ LLVM 降级
└── Target/LLVMIR/                    ★ LLVM IR 生成

# bishengir 端 (ascendnpu-ir)
ascendnpu-ir/bishengir/
├── include/bishengir/Dialect/        ★ HFusion/HIVM dialect
├── lib/Conversion/                   ★ 转换 Pass
│   ├── LinalgToHFusion/
│   ├── ArithToHFusion/
│   └── HFusionToHIVM/
└── tools/bishengir-opt/              ★ 编译入口
```
