# toy-mini — 从零写 Toy 语言解析器

纯 C++17 实现的 Toy 语言前端，零外部依赖。
对应 LLVM Toy Tutorial Ch1（Parser）+ Ch2（IR 生成）。

## 编译

```bash
cd projects/toy-mini
g++ -std=c++17 -o toymini toymini.cpp
./toymini
```

## 支持语法

```toy
# 函数定义
def main() {
  # 变量声明 + 二维数组字面量
  var A = [[1, 2, 3, 4], [5, 6, 7, 8]];

  # 转置
  var B = transpose(A);

  # 二元运算（+、-、*）
  var C = B + A;
  var D = C * 2.0;

  # 打印
  print(D);
  return;
}
```

## 组件

| 组件 | 行数 | 功能 |
|------|------|------|
| Lexer | ~150 | 14 token 类型，逐字符扫描 |
| Parser | ~400 | 递归下降，支持优先级控制 |
| AST | ~200 | 8 种节点，`toString()` 打印 |
| MLIR Gen | ~300 | 文本格式 IR 输出 |
| main | ~50 | 解析 + 打印 |

## 对照 Toy Tutorial

| 层面 | Toy Tutorial (LLVM) | Toy Mini |
|------|--------------------|----------|
| Lexer | `llvm::StringRef` | `std::string` |
| AST | `llvm::SmallVector` | `std::vector` |
| IR 生成 | TableGen ops (MLIR) | 文本模仿 |
| 编译 | LLVM 源码树内 | `g++ -std=c++17` |
| 依赖 | 全 LLVM 库 | 零 |

## 输出示例

```
=== MLIR 风格 IR ===
module {
  func.func @main() {
    %0 = toy.constant { dense<...> } : tensor<*xf64>
    %1 = toy.transpose(%0) : tensor<*xf64> -> tensor<*xf64>
    %2 = toy.add %1, %0 : tensor<*xf64>, tensor<*xf64> -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
```
