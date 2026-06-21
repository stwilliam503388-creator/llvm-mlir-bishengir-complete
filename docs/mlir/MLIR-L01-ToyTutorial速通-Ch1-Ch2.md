---

created: 2026-06-21
tags: [mlir, toy-tutorial, llvm, compiler, dialect]
aliases: [Toy Tutorial 速通, MLIR Toy Tutorial]

  - 工程
---

# MLIR Toy Tutorial 速通

> 基于 LLVM 22.1.6 官方 Toy Tutorial 源码 + bishengir 对照学习。
> 源码位置：`~/hermes-workspace/toy-tutorial/src/`

---

## 一、Toy 语言简介

Toy 是一个**教学用的数组语言**，类似 NumPy 的子集。

### 语法特征

```toy
# 定义一个函数
def multiply_transpose(A, B) {
  return transpose(A) * transpose(B);
}

# 支持数组字面量
def main() {
  var A = [[1, 2, 3, 4], [5, 6, 7, 8]];
  var B = [[1, 2, 3, 4], [5, 6, 7, 8]];
  var C = multiply_transpose(A, B);
  print(C);
}
```

### 支持的操作

| 操作 | 语法 | 说明 |
|------|------|------|
| **变量定义** | `var name = expr;` | 带类型推断 |
| **函数定义** | `def name(params) { body }` | 支持返回值 |
| **转置** | `transpose(A)` | 矩阵转置 |
| **逐元素加** | `A + B` | 矩阵加法 |
| **逐元素乘** | `A * B` | 逐元素乘法（非矩阵乘） |
| **标量乘** | `A * 2.0` | 广播到所有元素 |
| **打印** | `print(expr)` | 内置函数 |
| **返回** | `return expr;` | 函数返回值 |

### vs bishengir（ascendnpu-ir）对照

| 概念 | Toy | bishengir |
|------|-----|-----------|
| **语言前端** | Toy 语言 | Linalg dialect |
| **输入文件** | `.toy` | `.mlir` |
| **中间表示** | Toy dialect → MLIR | Linalg → HFusion → HIVM |
| **最终目标** | LLVM IR → CPU | HIVM IR → Ascend NPU |

---

## 二、Ch1：AST 解析器（纯 C++，无 MLIR）

### 架构

```
example.toy  →  Lexer (tokenize)  →  Parser (AST)  →  ASTPrinter (dump)
```

### 核心代码

**AST 节点类型**（`AST.h`）：

```cpp
// AST 中的一切节点
class ExprAST {                          // 表达式基类
  virtual ~ExprAST() = default;
  virtual llvm::Value *codegen() = 0;    // 后续章节的 MLIR codegen 入口
};

class NumberExprAST : ExprAST;           // 数字字面量: 1.0, 2.0
class VariableExprAST : ExprAST;         // 变量引用: A, B
class VarDeclExprAST : ExprAST;          // 变量声明: var name = expr
class BinaryExprAST : ExprAST;           // 二元操作: A + B, A * B
class CallExprAST : ExprAST;             // 函数调用: transpose(A), print(D)
class PrototypeAST : ExprAST;            // 函数原型: def name(params)
class FunctionAST : ExprAST;             // 函数定义: def name(params) { body }
class PrintExprAST : ExprAST;            // 打印语句: print(expr)
class ReturnExprAST : ExprAST;           // 返回语句: return expr
class LiteralExprAST : ExprAST;          // 数组字面量: [[1,2],[3,4]]
```

**toyc.cpp**（可执行文件入口）：

```cpp
int main(int argc, char **argv) {
  // 1. 读取 .toy 文件
  auto fileOrErr = llvm::MemoryBuffer::getFile(argv[1]);

  // 2. Lexer → Parser → AST
  Lexer lexer(fileOrErr.get());
  Parser parser(lexer);
  auto ast = parser.parseModule();       // 解析整个模块

  // 3. AST dump（仅 Ch1，Ch2 换成 MLIRGen）
  ModuleAST module = *ast;
  module.dump();                          // 打印 AST
}
```

### 解析过程

```
输入: def main() { var A = [[1,2],[3,4]]; print(A); }

Lexer 产出 token 流:
  def, main, (, ), {, var, A, =, [[, 1, ,, 2, ], [, 3, ,, 4, ]], ;, print, (, A, ), ;, }, EOF

Parser 构建 AST:
  ModuleAST
    └── FunctionAST "main"
          ├── PrototypeAST: name=main, params=[]
          └── VarDeclExprAST: name=A
                └── LiteralExprAST: [[1,2],[3,4]]
              PrintExprAST
                └── VariableExprAST: name=A
```

---

## 三、Ch2：定义 Toy Dialect + MLIR 生成 ← 核心章节

### 3.1 做了什么

Ch2 把 Ch1 的 AST dump 替换为 **MLIR 代码生成**。需要两件事：

1. **定义 Toy dialect**（用 TableGen 声明 ops 的类型签名）
2. **写 MLIRGen**（把 Toy AST 翻译成 MLIR IR）

### 3.2 架构

```
                   ┌─────────────────┐
  example.toy →    │   toyc (Ch2)    │    → MLIR IR (toy dialect ops)
                   │                 │
                   │  Parser → AST   │
                   │      ↓          │
                   │  MLIRGen        │
                   │      ↓          │
                   │  Toy Dialect    │  ← 来自 TableGen Ops.td
                   └─────────────────┘
```

### 3.3 TableGen 定义 Toy Dialect

**什么是 TableGen？**

LLVM 的声明式代码生成工具。写 `.td` 文件 → TableGen 编译时生成 `.inc` 文件（C++ 代码）。

**类比**：就像 protobuf 的 `.proto` → 生成 C++ 序列化代码。TableGen 是 LLVM 的"元编程"工具。

**Toy dialect 的 TableGen 定义**（`Ops.td`）：

```tablegen
// ops.td — Toy dialect 的操作定义
// 编译器看见这个文件 → 自动生成 C++ 代码

// Step 1: 定义 Toy dialect
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy language dialect for MLIR tutorial";
  let description = [{
    Toy 教学语言的 MLIR dialect。
    包含 transpose、print 等高级操作。
  }];
}

// Step 2: 定义 Toy 操作（Op）
// 每个操作就是一条 MLIR 指令

// 常量操作：定义一个张量常量
def ConstantOp : Toy_Op<"constant"> {
  let summary = "constant tensor";
  let arguments = (ins
    DenseFPElementsAttr:$value    // 常量值（稠密浮点数组）
  );
  let results = (outs
    Toy_TensorType:$output        // 输出张量
  );
  let assemblyFormat = "`{` $value `}` attr-dict `:` type($output)";
  // 文本格式: %0 = toy.constant { dense<1.0> : tensor<2x3xf64> } : tensor<2x3xf64>
}

// 加法操作
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition";
  let arguments = (ins
    Toy_TensorType:$lhs,
    Toy_TensorType:$rhs
  );
  let results = (outs
    Toy_TensorType:$result
  );
  // 自动推断 format: %r = toy.add %a, %b : tensor<2x3xf64>
}

// 转置操作
def TransposeOp : Toy_Op<"transpose"> {
  let arguments = (ins Toy_TensorType:$input);
  let results = (outs Toy_TensorType:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input) `->` type($output)";
  // 格式: %r = toy.transpose(%a) : tensor<2x3xf64> -> tensor<3x2xf64>
}

// 乘法操作（逐元素）
def MulOp : Toy_Op<"mul"> {
  let arguments = (ins Toy_TensorType:$lhs, Toy_TensorType:$rhs);
  let results = (outs Toy_TensorType:$result);
}

// 打印操作
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins Toy_TensorType:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

// 函数返回操作
def ReturnOp : Toy_Op<"return"> {
  let arguments = (ins Optional<Toy_TensorType>:$input);
}
```

**TableGen 生成的 C++ 代码**（等价于手写以下内容）：

```cpp
// TableGen 自动生成，你不用写
class ConstantOp : public mlir::Op<ConstantOp> {
  DenseFPElementsAttr getValue();
  TensorType getOutput();
  static void build(Builder &builder, OperationState &state,
                    DenseFPElementsAttr value, TensorType output);
};
```

### 3.4 MLIRGen：从 Toy AST 生成 MLIR IR

核心函数 `mlirGen()` 在 `MLIRGen.cpp`：

```cpp
class MLIRGenImpl {
public:
  // 入口：整个 Module → MLIR ModuleOp
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // 为每个顶层函数生成 MLIR
    for (FunctionAST &func : moduleAST)
      mlirGen(func);
  }

  // 函数定义 → MLIR func.func
  mlir::LogicalResult mlirGen(FunctionAST &funcAST) {
    // 1. 创建 func.func op
    // 2. 创建函数体 Region + Block
    // 3. 为每个 AST statement 生成 MLIR
    // 4. 设置 entry block 的参数
  }

  // 表达式 → MLIR Value（SSA 值）
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
      case ExprAST::Number:
        return mlirGen(cast<NumberExprAST>(expr));
      case ExprAST::Literal:
        return mlirGen(cast<LiteralExprAST>(expr));
      case ExprAST::Variable:
        return mlirGen(cast<VariableExprAST>(expr));
      case ExprAST::Binary:
        return mlirGen(cast<BinaryExprAST>(expr));
      case ExprAST::Call:
        return mlirGen(cast<CallExprAST>(expr));
      // ...
    }
  }
};
```

### 3.5 MLIRGen 实战：`example.toy` 的生成结果

**输入** `example.toy`：
```toy
def main() {
  var A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
  var B = transpose(A);
  var C = B + A;
  var D = C * 2.0;
  print(D);
}
```

**MLIRGen 输出**（生成的 Toy dialect IR）：

```mlir
module {
  func.func @main() {
    // var A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    %0 = toy.constant { dense<[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]> : tensor<4x4xf64> } : tensor<4x4xf64>

    // var B = transpose(A)
    %1 = toy.transpose(%0) : tensor<4x4xf64> -> tensor<4x4xf64>

    // var C = B + A
    %2 = toy.add %1, %0 : tensor<4x4xf64>

    // var D = C * 2.0   (2.0 先变成常量，再做 mul)
    %3 = toy.constant { dense<2.0> : tensor<f64> } : tensor<f64>
    %4 = toy.mul %2, %3 : tensor<4x4xf64>

    // print(D)
    toy.print(%4) : tensor<4x4xf64>

    // return
    toy.return
  }
}
```

### 3.6 MLIRGen 逐行解读

以 `var A = [[1,2,3,4], ...]` 为例：

```cpp
// 1. 解析器先识别 LiteralExprAST
// LiteralExprAST 包含:
//   dims = [4, 4]  ← 从 [[...]] 嵌套层级推断
//   values = [1, 2, ..., 16]  ← 展开的所有值

// 2. MLIRGen 生成 MLIR 常量操作
mlir::Value MLIRGenImpl::mlirGen(LiteralExprAST &lit) {
  // 获取张量维度
  auto shape = lit.getDims();  // {4, 4}

  // 创建 MLIR 类型: tensor<4x4xf64>
  auto type = Toy_TensorType::get(builder, shape);

  // 创建 DenseFPElementsAttr: 存储常量值
  auto data = DenseFPElementsAttr::get(type, lit.getValues());

  // 创建 toy.constant 操作
  auto op = builder.create<ConstantOp>(loc, type, data);

  // 返回 SSA 值（供后续操作引用）
  return op.getResult();
}
```

### 3.7 Toy dialect 的语义

| Toy Op | 语法 | 语义 |
|--------|------|------|
| `toy.constant` | `%c = toy.constant { dense<1.0> } : tensor<2x3xf64>` | 创建一个常量张量 |
| `toy.add` | `%r = toy.add %a, %b : tensor<2x3xf64>` | 逐元素加法 |
| `toy.mul` | `%r = toy.mul %a, %b : tensor<2x3xf64>` | 逐元素乘法 |
| `toy.transpose` | `%r = toy.transpose(%a) : tensor<4x4xf64> -> tensor<4x4xf64>` | 矩阵转置 |
| `toy.print` | `toy.print(%a) : tensor<2x3xf64>` | 打印张量到 stdout |
| `toy.return` | `toy.return` 或 `toy.return %val` | 函数返回 |

---

## 四、与 bishengir 对照

| 概念 | Toy Tutorial | bishengir (ascendnpu-ir) |
|------|-------------|--------------------------|
| **自定义 dialect** | `toy` dialect | `hfusion` + `hivm` + 6 个其他 dialect |
| **TableGen 定义 ops** | `Ops.td` | `HFusionStructuredOps.td`, `HIVMVectorOps.td` |
| **Dialect 生成方式** | `mlir-tblgen` 编译时生成 | 同样用 `mlir-tblgen` |
| **Op 的参数声明** | `(ins Toy_TensorType, outs Toy_TensorType)` | `(ins AnyType, AnyType, BinaryFnAttr)` |
| **MLIR 生成** | Toy AST → Toy dialect | Triton TIR → HFusion dialect |
| **转换** | Ch5 开始 lower 到标准 dialect | HFusion → HIVM |
| **最终目标** | LLVM IR → CPU | HIVM IR → Ascend NPU |

Toy Tutorial 学完，bishengir 的 dialect 定义和 pass 体系**本质是一回事**，只是 ops 不一样、dialect 名字不同。
