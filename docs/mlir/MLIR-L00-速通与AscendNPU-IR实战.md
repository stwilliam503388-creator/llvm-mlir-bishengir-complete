> 📍 Phase 3 MLIR | [返回入口](./README.md)
> 前置：[00-从LLVM到MLIR](./00-从LLVM到MLIR.md)
> 预估时间：30 min

---
created: 2026-06-21
tags: [mlir, llvm, compiler, ascendnpu-ir, learning]
aliases: [MLIR 入门, MLIR 速通]
---

# MLIR 速通与 AscendNPU-IR 实战

> LLVM 速通的续篇，从 LLVM IR 进入 MLIR 世界。
> 基于 AscendNPU-IR（ascendnpu-ir）的 dialect 和 Pass 源码，**用实际代码讲概念**。

---

## 一、MLIR 到底是什么？

### 一句话

**MLIR（Multi-Level Intermediate Representation）** 是 LLVM 项目下的多层中间表示框架——它允许编译器在 **多个抽象层级** 上表达和优化代码，而不是像 LLVM IR 那样只有一个固定的中间层。

### LLVM IR vs MLIR

| 维度 | LLVM IR | MLIR |
|------|---------|------|
| **抽象层级** | 固定（接近汇编） | **多级**（从高级算子→低级指令） |
| **类型系统** | 标量 + 指针 + 向量 | **任意类型**（tensor、memref、自定义） |
| **操作表示** | 有限的指令集 | **可扩展的操作集**（dialect） |
| **优化方式** | LLVM Pass（固定操作） | **Dialect-specific Pass**（自定义操作专用优化） |
| **适用场景** | 通用编译后端 | 领域特定编译器（ML/DL/GPU/NPU） |

### 为什么 Ascend NPU 用 MLIR 而不是 LLVM IR？

```
C 代码         → LLVM IR        → x86 / ARM 指令
            （适用于通用 CPU）

Triton 代码   → TIR (MLIR)     → AIR (MLIR)     → HIVM IR → NPU 指令
            （适用于 NPU/GPU 专用硬件）

昇腾的流水线：   linalg dialect  → hfusion dialect → hivm dialect → 机器码
                                          ↑
                                    MLIR 的多级 dialect 体系正是为此而生
```

**核心优势**：NPU 有特殊的硬件单元（Cube 矩阵乘、Vector 向量处理、Scalar 标量处理），MLIR dialect 可以为每一级硬件抽象层定义专属操作，然后逐步降级到目标指令。

---

## 二、MLIR 核心概念

### 2.1 Operation（操作）

MLIR 中的一切**都是操作（Operation）**。包括函数定义、变量声明、控制流、计算指令。

```mlir
%result = arith.addf %a, %b : f16
│        │      │      │     │
│        │      │      │     └── 类型（f16 浮点）
│        │      │      └──────── 操作数（SSA 值）
│        │      └─────────────── 操作名（addf = add float）
│        └───────────────────── dialect 名（arith = arithmetic）
└────────────────────────────── SSA 结果
```

**与 LLVM IR 的对照：**

| LLVM IR | MLIR 等价 | 说明 |
|---------|----------|------|
| `%r = add i32 %a, %b` | `%r = arith.addi %a, %b : i32` | 整数加法 |
| `%r = fadd float %a, %b` | `%r = arith.addf %a, %b : f32` | 浮点加法 |
| `call void @foo()` | `func.call @foo() : () -> ()` | 函数调用 |
| `define @f(i32) i32` | `func.func @f(%arg0: i32) -> i32` | 函数定义 |
| 无直接等价 | `linalg.generic {...}` | 线性代数操作 |
| 无直接等价 | `tensor<4x4xf16>` | 张量类型 |

### 2.2 Dialect（方言）

Dialect 是一组相关的操作、类型和属性的**命名空间集合**。

```
arith.addf     ← arith dialect（算术操作）
scf.for        ← scf dialect（结构化控制流）
linalg.matmul  ← linalg dialect（线性代数）
hivm.vadd      ← hivm dialect（HIVM 向量指令，AscendNPU-IR 定义）
```

**标准 dialect（随 MLIR 发布）：**

| Dialect | 用途 | 类比 |
|---------|------|------|
| `builtin` | 内置类型（i32, f32）、module、func | C 语言基本类型 |
| `arith` | 加减乘除、比较、类型转换 | CPU 算术指令 |
| `math` | sqrt、sin、cos 等数学函数 | `<math.h>` |
| `scf`（Structured Control Flow） | for 循环、while 循环、if-else | 结构化控制流 |
| `cf`（Control Flow） | 无条件/条件跳转 | goto |
| `linalg`（Linear Algebra） | matmul、conv、generic 操作 | BLAS 库 |
| `tensor` | 张量创建、提取、拼接 | NumPy |
| `memref` | 内存引用、加载、存储 | C 指针 |
| `func` | 函数定义和调用 | C 函数 |
| `affine` | 仿射循环、仿射内存访问 | 多面体模型 |

**AscendNPU-IR 自定义 dialect：**

| Dialect | 命名空间 | 用途 |
|---------|---------|------|
| `HFusion` | `hfusion.*` | 华为融合算子层（fused element-wise ops） |
| `HIVM` | `hivm.*` | 华为 IR for Vector & Matrix（NPU 指令层） |
| 其他 6 个 | — | 其他 AscendNPU-IR dialect（总数 8 个） |

### 2.3 Region 与 Block

MLIR 比 LLVM IR 多了一层**嵌套结构**：

```
module {                          ← Region（顶层）
  func.func @vecadd(...) {        ← Region（函数体）
    ^bb0:                         ← Block（基本块）
      linalg.generic {            ← Operation（含 Region）
        ^bb0(%a: f16, ...):       ← Block（linalg body）
          ...
      }
      return
  }
}
```

**关键差异：** LLVM IR 只能在基本块层面嵌套；MLIR 可以在**任何操作内部**嵌套 Region + Block。这让 MLIR 能表达 `linalg.generic` body、`scf.for` 循环体等结构化控制流。

### 2.4 类型系统

MLIR 类型远比 LLVM IR 丰富：

| 类型 | 示例 | 说明 |
|------|------|------|
| **iN** | `i32`, `i1` | 同 LLVM IR |
| **fN** | `f16`, `f32`, `f64` | 浮点 |
| **index** | `index` | 架构无关的索引类型 |
| **tensor** | `tensor<4x4xf16>`, `tensor<?xf32>` | 张量（可能有 SSA 静态形状或动态 `?`） |
| **memref** | `memref<1024xf16, affine_layout>` | 带布局信息的内存引用 |
| **vector** | `vector<4xf32>` | SIMD 向量 |
| **tuple** | `tuple<i32, f32>` | 元组 |
| **none** | `none` | 无类型（标记用） |
| **自定义** | (dialect 可定义) | 如 HIVM dialect 的自定义类型 |

**tensor vs memref 的区别：**

| 概念 | 位置 | 用途 |
|------|------|------|
| **tensor** | SSA 值（寄存器/L1） | 计算过程中的数据 |
| **memref** | 内存（显存/主存） | 输入输出缓冲区，有指针语义 |

```
memref<4x4xf16>     —— 在显存/主存中
     │
  hivm.load          —— 加载到 L1 Buffer
     │
tensor<4x4xf16>     —— 在 L1 Buffer（计算域）
     │
  hivm.vadd          —— NPU Vector 单元计算
     │
  output tensor      —— 计算结果
     │
  hivm.store         —— 写回显存
     │
memref<4x4xf16>     —— 回到显存
```

---

## 三、MLIR Pass 体系

### 3.1 与 LLVM Pass 的对比

| 维度 | LLVM Pass | MLIR Pass |
|------|-----------|-----------|
| **操作目标** | LLVM IR（固定指令集） | **任意 MLIR 操作**（dialect 无限制） |
| **匹配方式** | 指令遍历 + 迭代 | **Pattern 匹配**（RewritePattern） |
| **注册方式** | `INITIALIZE_PASS` / `PassInfoMixin` | `mlir::RewritePatternSet` 注册 |
| **常见类型** | FunctionPass / ModulePass | **OperationPass<T>**（对具体某个操作生效） |
| **转换框架** | 手工写 loop | **Dialect Conversion 框架**（自动处理 legality/type conversion） |
| **新 Pass 写法** | New PM（`run` 方法） | `runOnOperation()` 方法 |

### 3.2 OperationPass 基础

```cpp
// MLIR Pass 的通用写法
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
    // 对每个 func.func 执行一次
    void runOnOperation() override {
        func::FuncOp func = getOperation();
        // ... 遍历 func body 内的操作，做转换 ...
    }
    
    // 必须声明（供 MLIR 框架反射）
    StringRef getArgument() const override { return "my-pass"; }
    StringRef getDescription() const override { return "does something"; }
};
```

### 3.3 MLIR 的 Pattern 匹配

```cpp
// 方法1: 在 runOnOperation 内用 walk() 手动匹配 ops
void runOnOperation() override {
    getOperation()->walk([](Operation* op) {
        if (auto add = dyn_cast<arith::AddFOp>(op)) {
            // 处理每个 addf
        }
    });
}

// 方法2: 用 RewritePattern（推荐）
struct MyRewritePattern : public OpRewritePattern<arith::AddFOp> {
    LogicalResult matchAndRewrite(arith::AddFOp op, PatternRewriter& rewriter) const override {
        // match: 检查 AddFOp 是否符合条件
        // rewrite: 用 rewriter 创建新操作替换旧的
        rewriter.replaceOpWithNewOp<MyCustomOp>(op, op.getLhs(), op.getRhs());
        return success();
    }
};
```

### 3.4 Dialect Conversion 框架（AscendNPU-IR 用的就是这个）

这是 MLIR 最强大的机制——用于**跨 dialect 的完整转换**：

```
Linalg IR  ──→  HFusion IR  ──→  HIVM IR
    ↑              ↑                 ↑
ConvertLinalg    ConvertArith     ConvertHFusion
ToHFusion()      ToHFusion()      ToHIVM()

每个转换包含：
  1. Target（合法性定义）—— 哪些操作是"合法"（目标 dialect 的）
  2. Pattern（转换规则） —— 如何把源操作替换为目标操作
  3. TypeConverter（类型转换）—— memref → tensor 等
```

**AscendNPU-IR 的 ConvertLinalgToHFusion 示例：**

```cpp
// 文件: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
// 作用: 把 linalg.generic + arith.addf 合并为 hfusion.elemwise_binary

// Step 1: 定义目标（目标 dialect 中的合法操作）
void ConvertLinalgToHFusion::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<HFusionDialect>();   // 只允许 HFusion 操作
    target.addIllegalDialect<linalg::LinalgDialect>();  // 禁止 Linalg 操作

    RewritePatternSet patterns(&getContext());
    
    // Step 2: 注册匹配规则
    patterns.add<ConvertLinalgGenericToHFusion>(&getContext());
    
    // Step 3: 执行整体转换（涉及类型转换、区域转换）
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}
```

---

## 四、AscendNPU-IR 三阶段降级实战（代码解读）

### 4.1 阶段一：Linalg → HFusion

**输入**（test file: `linalg-to-hfusion.mlir`）：
```mlir
// vecadd 的 linalg.generic 表达
func.func @vecadd(%A: memref<1024xf16>, %B: memref<1024xf16>, %C: memref<1024xf16>) {
  linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : memref<1024xf16>, memref<1024xf16>)
    outs(%C : memref<1024xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  }
  return
}
```

**经过 `-convert-linalg-to-hfusion -convert-arith-to-hfusion`：**
```mlir
// 输出：hfusion.elemwise_binary 操作
// 把 linalg.generic body 中的 arith.addf 识别为"二元加法"
func.func @vecadd(%A: memref<1024xf16>, %B: memref<1024xf16>, %C: memref<1024xf16>) {
  %result = hfusion.elemwise_binary {
    fun = #hfusion.binary_fn<add>     // 枚举：加法类型
  } ins(%A, %B : memref<1024xf16>, memref<1024xf16>)
    outs(%C : memref<1024xf16>) : memref<1024xf16>
  return
}
```

**发生了什么（从 LinalgToHFusion.cpp）：**

```
linalg.generic {
  ^bb0:
    ↓  arith.addf
    ↓  linalg.yield
}
         ↓
Pattern 匹配器识别：
  - body 中只有 1 个 arith.addf → binary_fn<add>
  - body 中只有 1 个 arith.subf → binary_fn<sub>
  - body 中只有 1 个 arith.mulf → binary_fn<mul>
  - body 中有 arith.addf + arith.subf → 复杂，不合并

         ↓
hfusion.elemwise_binary {fun = add}
```

### 4.2 阶段二：Arith → HFusion

**作用：** 把没有被 Linalg 转换的独立 `arith.*` 操作也转为 HFusion

```mlir
// 转换前：
%sum = arith.addf %a, %b : f16

// 转换后（内部表示，前面 Linalg 阶段已合并）
// 如果还有孤立的 arith，会被 ArithToHFusion 处理
```

### 4.3 阶段三：HFusion → HIVM

**输入**（test file: `hfusion-to-hivm.mlir`）：
```mlir
// hfusion.elemwise_binary 操作
%result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<add>}
  ins(%A, %B : memref<1024xf16>, memref<1024xf16>)
  outs(%C : memref<1024xf16>) : memref<1024xf16>
```

**经过 `-convert-hfusion-to-hivm`：**
```mlir
// 输出：HIVM 指令——load → vadd → store
func.func @vecadd(%A: memref<1024xf16>, %B: memref<1024xf16>, %C: memref<1024xf16>) {
  // Step 1: 从显存加载到 L1 Buffer（memref → tensor）
  %tA = hivm.load %A : memref<1024xf16> -> tensor<1024xf16>
  %tB = hivm.load %B : memref<1024xf16> -> tensor<1024xf16>

  // Step 2: 核心运算——NPU Vector 单元执行向量加法
  %sum = hivm.vadd %tA, %tB : tensor<1024xf16>

  // Step 3: 写回显存（tensor → memref）
  hivm.store %sum, %C : tensor<1024xf16> -> memref<1024xf16>
  return
}
```

**完整三阶段 pipeline：**

```
vecadd_input.mlir
    │
    ├── ─convert-linalg-to-hfusion    [LinalgToHFusion.cpp]
    │   linalg.generic{arith.addf} → hfusion.elemwise_binary{add}
    │
    ├── ─convert-arith-to-hfusion     [ArithToHFusion.cpp]
    │   处理剩余未被合并的 arith 操作
    │
    └── ─convert-hfusion-to-hivm      [HFusionToHIVM.cpp]
        hfusion.elemwise_binary{add} → hivm.load + hivm.vadd + hivm.store

最终：hivm IR（NPU 指令级）
```

---

## 五、实操：用 Homebrew mlir-opt 体验 MLIR

AscendNPU-IR 需要 Ascend NPU SDK 才能完全编译，但标准 MLIR 的 `linalg` → `affine` → `scf` → `llvm` 流水线在你的 Mac 上可以直接跑。

### 5.1 设置环境

```bash
export LLVM_DIR="/opt/homebrew/opt/llvm"
export PATH="$LLVM_DIR/bin:$PATH"
```

### 5.2 写一个 VecAdd 的 MLIR 输入

```mlir
// vecadd.mlir — 与 AscendNPU-IR linalg-to-hfusion.mlir 完全相同的结构
module {
  func.func @vecadd(%A: memref<1024xf16>, %B: memref<1024xf16>, %C: memref<1024xf16>) {
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : memref<1024xf16>, memref<1024xf16>)
      outs(%C : memref<1024xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    }
    return
  }
}
```

### 5.3 完整降级流水线

```bash
mlir-opt \
  --convert-linalg-to-affine-loops \    # linalg → affine（展开为显式循环）
  --lower-affine \                       # affine → scf（控制流结构化）
  --convert-scf-to-cf \                  # scf → cf（goto 风格）
  --convert-func-to-llvm \               # func → LLVM IR
  --convert-arith-to-llvm \              # arith → LLVM IR
  vecadd.mlir
```

**输出效果：**

```
// 最终 LLVM IR — 与上面 AscendNPU-IR 流水线做概念对比
llvm.func @vecadd(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {
  // 无显式 load/store — 由 memref 隐式处理
  // 无显式 vadd 指令 — 由通用 CPU 算术指令表示
  // 优化器后续做 auto-vectorization
}
```

**对比 AscendNPU-IR 流水线：**

| 阶段 | MLIR 标准流水线 | AscendNPU-IR 流水线 |
|------|----------------|-----------------|
| 高级 | `linalg.generic` | `linalg.generic` |
| 中级 | `affine.for` / `scf.for` | `hfusion.elemwise_binary` |
| 低级 | `cf.br` / LLVM IR | `hivm.load` / `hivm.vadd` / `hivm.store` |
| 最终 | CPU 指令 | NPU 指令（通过 CANN 后端） |
| **优化目标** | 通用 CPU 向量化 | NPU Vector/Cube 单元 |

---

## 六、TableGen 定义解读（AscendNPU-IR 的 dialect 定义）

### 6.1 `HFusionStructuredOps.td`

位置：`bishengir/include/.../Dialect/HFusion/IR/HFusionStructuredOps.td`

```tablegen
// 定义 HFusion dialect
def HFusion_Dialect : Dialect {
  let name = "hfusion";
  let summary = "HFusion 算子融合方言";
}

// 定义逐元素二元操作
def HFusion_ElemwiseBinaryOp : HFusion_Op<"elemwise_binary"> {
  let summary = "Element-wise binary operation";
  let description = [{
    逐元素二元操作（add/sub/mul/div 等）。
    用于表示诸如 a + b、a * b 之类每个元素独立计算的运算。
  }];

  // 输入参数
  let arguments = (ins
    AnyType:$lhs,        // 左操作数 A[i]
    AnyType:$rhs,        // 右操作数 B[i]
    BinaryFnAttr:$fun    // 运算类型枚举（add/sub/mul/div/max/min/...）
  );
  
  // 输出结果
  let results = (outs AnyType:$result);
  
  // 汇编格式
  let assemblyFormat = [{
    `{` `fun` `=` $fun `}` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` 
    `outs` `(` $result `:` type($result) `)` `:` type($result)
  }];
}
```

### 6.2 `HIVMVectorOps.td`

位置：`bishengir/include/.../Dialect/HIVM/IR/HIVMVectorOps.td`

```tablegen
// 定义 HIVM dialect（NPU 指令级）
def HIVM_Dialect : Dialect {
  let name = "hivm";
  let summary = "Huawei Intermediate Vector Machine Dialect";
}

// 向量加载指令：memref → tensor（显存 → L1 Buffer）
def HIVM_LoadOp : HIVM_Op<"load"> {
  let summary = "Load from memory to register";
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs AnyRankedTensor:$result);
}

// 向量加法指令（核心运算）
def HIVM_VAddOp : HIVM_Op<"vadd"> {
  let summary = "Vector ADD — element-wise addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

// 向量存储指令：tensor → memref（L1 Buffer → 显存）
def HIVM_StoreOp : HIVM_Op<"store"> {
  let summary = "Store from register to memory";
  let arguments = (ins AnyRankedTensor:$value, AnyMemRef:$memref);
}
```

### 6.3 TableGen 速查

| TableGen 语法 | 作用 | 类比 |
|---------------|------|------|
| `def Foo` | 定义一个记录（Record） | 定义一个类 |
| `def Foo : Dialect` | 定义一个 dialect | 定义一个命名空间 |
| `def FooOp : HIVM_Op<"name">` | 定义一个操作 | 定义一条指令 |
| `let arguments = (ins ...)` | 声明输入参数 | 指令的输入操作数 |
| `let results = (outs ...)` | 声明输出参数 | 指令的输出操作数 |
| `let assemblyFormat = [...]` | 声明文本格式 | 语法规则 |
| `let summary = "..."` | 简短描述 | 文档 |
| `let description = [{...}]` | 详细描述 | 文档 |
| `ins` / `outs` | 类型约束中的输入/输出标记 | |
| `AnyType` / `AnyMemRef` / `AnyRankedTensor` | 类型约束 | 泛型类型匹配 |

---

## 七、关键概念对照表

### LLVM IR → MLIR 对照

| LLVM IR 概念 | MLIR 概念 | 区别 |
|-------------|-----------|------|
| `Function` | `func.func` (Operation) | MLIR 的 func.func 本身也是一个操作 |
| `BasicBlock` | `Block` | 类似，但 MLIR block 可嵌套 |
| `Instruction` | `Operation` | MLIR 操作更灵活（可嵌套 Region） |
| `Type` | `Type` | MLIR 类型更丰富（tensor/memref） |
| `Value` | `Value` | SSA 值，用法类似 |
| `Module` | `module` (内置 Op) | 类似 |
| `Pass` | `Pass` / `RewritePattern` | MLIR 有更强大的转换框架 |
| `Metadata` | `Attribute` / `DictionaryAttr` | 类似，MLIR 更结构化 |
| LLVM Pass | MLIR Dialect Conversion | MLIR 的跨 dialect 转换是独特能力 |

### AscendNPU-IR 三阶段关键操作对照

| 操作 | Dialect | 层级 | 含义 |
|------|---------|------|------|
| `linalg.generic` | `linalg` | 高级 | 通用线性代数操作 |
| `arith.addf` | `arith` | 高级 | 浮点加法 |
| `hfusion.elemwise_binary` | `hfusion` | 中级 | 华为融合算子 |
| `hivm.load` | `hivm` | 低级 | 显存→L1 Buffer 加载 |
| `hivm.vadd` | `hivm` | 低级 | NPU 向量加法指令 |
| `hivm.store` | `hivm` | 低级 | L1 Buffer→显存写回 |

---

## 八、下一步方向

| 方向 | 内容 | 可行性 |
|------|------|--------|
| **bishengir-opt 编译** | 初始化 LLVM 子模块，全量编译 | ⚠️ 耗时 1-2h，需要 ~30GB 磁盘，可后台跑 |
| **写自定义 MLIR Pass** | 针对 AscendNPU-IR dialect 写简单的 pattern match Pass | ✅ 无需编译，写代码分析即可 |
| **MLIR Toy Tutorial** | 官方教程：定义自己的 dialect + lower 到 LLVM IR | ✅ 可编译运行（用 Homebrew MLIR）|
| **VecAdd 完整驱动** | 写一个 main.cpp 加载 .mlir 文件跑 Pass pipeline | ✅ 需要 Homebrew MLIR 链接 |
| **Ascend 实机测试** | 在有 Ascend NPU 的机器上编译运行 | ❌ 目前无 NPU 硬件 |

---

## 附：关键文件位置

```text
ascendnpu-ir/
├── bishengir/
│   ├── include/bishengir/Dialect/
│   │   ├── HFusion/IR/HFusionStructuredOps.td    ★ HFusion dialect 定义
│   │   └── HIVM/IR/HIVMVectorOps.td               ★ HIVM dialect 定义
│   ├── lib/Conversion/
│   │   ├── LinalgToHFusion/LinalgToHFusion.cpp    ★ Linalg→HFusion Pass
│   │   ├── ArithToHFusion/ArithToHFusion.cpp      ★ Arith→HFusion Pass
│   │   └── HFusionToHIVM/HFusionToHIVM.cpp        ★ HFusion→HIVM Pass
│   ├── test/Conversion/
│   │   ├── LinalgToHFusion/linalg-to-hfusion.mlir ★ 转换测试（输入+期望输出）
│   │   └── HFusionToHIVM/hfusion-to-hivm.mlir     ★ 转换测试
│   └── tools/bishengir-opt/
│       └── bishengir-opt.cpp                       ★ 主入口
└── build-tools/build.sh                            ★ 编译脚本
```

---

## 九、术语速查

| 术语 | 中文 | 一句话解释 |
|------|------|-----------|
| **MLIR** | 多层中间表示 | 允许编译器在多个抽象层级上表达和优化代码的框架 |
| **Dialect** | 方言 | 一组相关的操作、类型和属性的命名空间集合 |
| **Operation** | 操作 | MLIR 中的一切指令（函数、运算、类型转换都是操作）|
| **SSA** | 静态单赋值 | 每个变量被赋值一次且只能一次 |
| **Region** | 区域 | 操作内部的嵌套 IR 块（允许递归结构）|
| **Block** | 基本块 | 一个顺序的操作序列，末尾可以有跳转 |
| **Pattern** | 匹配模式 | 定义如何匹配和重写 MLIR 操作的规则 |
| **Dialect Conversion** | 方言转换 | 把源 dialect 操作整体转换到目标 dialect 的框架 |
| **TableGen** | 表格生成器 | LLVM 的声明式代码生成工具（给 .td 文件用的）|
| **Linalg** | 线性代数 | MLIR 的标准线性代数 dialect |
| **HFusion** | 华为融合层 | AscendNPU-IR 的中间 dialect，做算子融合 |
| **HIVM** | 华为向量机 | AscendNPU-IR 的低级 dialect，NPU 指令级 |
| **memref** | 内存引用 | 带布局信息的缓冲区引用（可以指显存地址）|
| **tensor** | 张量 | 多维数组值（在计算域中，与硬件无关）|
| **CANN** | 昇腾计算框架 | Huawei 的 NPU SDK（包含了 blas/加速库）|


> 📖 [术语表](../glossary.md)