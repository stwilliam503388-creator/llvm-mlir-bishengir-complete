> 📍 Phase 3 MLIR | [返回入口](./README.md)
> 前置：[00-从LLVM到MLIR](./00-从LLVM到MLIR.md)
> 预估时间：15 min

---
created: 2026-06-21
tags: [toy, mini, parser, learning]
aliases: [Toy Mini 语言, 从零写 Toy 解析器]
---

# Toy Mini 语言：从零实现

> 手写 Toy 语言的前端（Lexer + Parser + AST + IR 生成）。
> 纯 C++17，零依赖。对应 Toy Tutorial Ch1 + Ch2。

---

## 项目概况

位置：`~/hermes-workspace/toy-tutorial/toymini/toymini.cpp`

已验证：✅ C++17 编译通过（0 errors）

### 包含的组件

| 组件 | 功能 | 对应 Toy Tutorial |
|------|------|------------------|
| **Lexer** | 词法分析：tokenize 源码 | Ch1 |
| **Parser** | 语法分析：递归下降解析 | Ch1 |
| **AST** | 语法树节点打印 | Ch1 |
| **MLIR Gen** | MLIR 风格 IR 文本输出 | Ch2 |

### Toy Mini 语法支持

```toy
# 函数定义
def main() {
  # 变量声明 + 数组字面量
  var A = [[1, 2, 3, 4], [5, 6, 7, 8]];
  var B = transpose(A);

  # 二元运算
  var C = B + A;
  var D = C * 2.0;

  # 打印
  print(D);
  return;
}
```

所有关键字：`def`, `var`, `return`, `print`, `transpose`
所有运算符：`+`, `-`, `*`
字面量：数字（整数/浮点）、二维数组 `[[...]]`
注释：`#` 开头到行末

---

## 核心架构

### Lexer（词法分析器）

```cpp
class Lexer {
  // 逐个字符扫描，按模式识别 token
  // 支持：关键字优先（def/var/return 等）
  //       数字（整数+浮点）
  //       标识符（字母开头）
  //       注释（# 到行尾）
};
```

Token 类型：14 种（LParen, RParen, Number, Ident, Def, Var, ...）

### Parser（递归下降）

```
parseModule()
  └── parseFunction()     — def name(params) { body }
       ├── parseStatement() — var / print / return / expr
       │    ├── parseExpr()     — 表达式（优先级控制）
       │    │    ├── parseBinary()  — 二元运算（+/* 优先级）
       │    │    └── parsePrimary() — 数字/标识符/调用/字面量
       │    │         ├── parseLiteral() — [[1,2],[3,4]]
       │    │         └── 函数调用: name(...)
       │    └── 变量声明: var name = expr
       └── ...
```

### MLIR 生成（文本格式）

```mlir
module {
  func.func @main() {
    %0 = toy.constant { dense<...> } : tensor<*xf64>
    // ... ops ...
    toy.return
  }
}
```

---

## 与 Toy Tutorial 对照

| 特性 | Toy Tutorial (LLVM) | Toy Mini (纯手写) |
|------|--------------------|-------------------|
| Lexer | LLVM 的 `llvm::StringRef` | 纯 `std::string` |
| Parser | 递归下降 | 递归下降 |
| AST | LLVM 的 `std::unique_ptr` + `llvm::SmallVector` | 纯 `std::unique_ptr` + `std::vector` |
| IR 生成 | 真实 MLIR (TableGen Ops) | 文本模仿 MLIR |
| 编译 | 需要 LLVM 源码树 | g++ -std=c++17 搞定 |

---

## 编译运行

```bash
cd ~/hermes-workspace/toy-tutorial/toymini
g++ -std=c++17 -o toymini toymini.cpp
./toymini
```

输出：
```
=== 源码 ===
def main() { ... }

=== AST ===
Function(main)
  VarDecl(A)
    Literal[2x4]
  VarDecl(B)
    Call(transpose)
  VarDecl(C)
    Binary(+)
  Print
  Return

=== MLIR 风格 IR ===
module {
  func.func @main() {
    %0 = toy.constant { ... } : tensor<*xf64>
    // ...
    toy.return
  }
}
```


> 📖 [术语表](../glossary.md)