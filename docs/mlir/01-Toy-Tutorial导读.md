# 01 — Toy Tutorial 导读

> 目标：跑通 MLIR 官方 Toy Tutorial 的第 1-3 章
> 前置：[00 — 从 LLVM 到 MLIR](./00-从LLVM到MLIR.md)
> 预估时间：60 分钟（30 读 + 30 跑代码）

## 1. Toy 语言是什么？

Toy 是 MLIR 官方为教学设计的一个极简语言。只做一件事：**张量计算**。

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

为什么选张量计算？因为所有 AI 框架本质上都在做张量计算。Toy 是 AI 编译器的微缩模型。

## 2. 环境确认

```bash
mlir-opt --version   # brew install llvm 已自带
```

## 3. 三章带读

### Ch1：定义 Toy 语言的 AST

**在干什么**：写一个解析器，把 `.toy` 源码变成 AST。

和编译原理课的 Lex/Yacc 一样。Toy 教程已写好，**不需要自己写**。

→ 详细代码见 [MLIR-L01](./MLIR-L01-ToyTutorial速通-Ch1-Ch2.md)

### Ch2：用 MLIR 表示 Toy 程序（关键章节）

**在干什么**：定义 `toy` Dialect，把 Toy 程序翻译成 MLIR。

你会看到：
```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) {
  %0 = toy.transpose(%arg0) to tensor<*xf64>
  %1 = toy.transpose(%arg1) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

对比 LLVM IR，MLIR 的好处很明显：`toy.transpose` — 一眼看出在做什么。

→ 详细代码见 [MLIR-L01](./MLIR-L01-ToyTutorial速通-Ch1-Ch2.md)

### Ch3：写 Pass 优化 Toy IR

**在干什么**：消除冗余转置。`transpose(transpose(x)) = x`。

```mlir
// 优化前：两次转置
%0 = toy.transpose(%arg0)
%1 = toy.transpose(%0)        ← 多余！
%2 = toy.mul %arg1, %1

// 优化后：直接乘
%0 = toy.mul %arg1, %arg0
```

**这就是 Pass**——和你写的 HelloPass 思路一样：遍历 IR → 匹配模式 → 替换。

→ 详细代码见 [MLIR-L02](./MLIR-L02-ToyTutorial速通-Ch3-Ch6.md)

## 4. 和 HelloPass / mlir-hello 的对比

| | hello-pass | mlir-hello | Toy Ch3 Pass |
|---|---|---|---|
| 遍历单位 | `Function` | `func::FuncOp` | `Operation` |
| 做什么 | 打印信息 | 打印信息 | 消除冗余操作 |
| 修改 IR | ❌ | ❌ | ✅ |

## 5. 常见坑

| 问题 | 解决 |
|------|------|
| 找不到 mlir-opt | `export PATH="/opt/homebrew/opt/llvm/bin:$PATH"` |
| Toy 构建报错 | 用 LLVM 源码 `mlir/examples/toy/` 直接 cmake |
| .mlir 文件看不懂 | 回到 [00-从LLVM到MLIR](./00-从LLVM到MLIR.md) |

## 验证

- [ ] 能说出 Toy 语言是干什么的
- [ ] 能指出 `.mlir` 中 `toy.transpose` 的 Dialect
- [ ] Ch3 的 Pass 和 mlir-hello 有什么相同和不同

> 📖 [术语表](../glossary.md)
> **下一步**：[02 — ascendnpu-ir 快速上手](./02-ascendnpu-ir快速上手.md)
