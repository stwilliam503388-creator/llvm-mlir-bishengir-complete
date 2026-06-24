# HelloMLIRPass — 你的第一个 MLIR Pass

MLIR 版 HelloPass，和 Phase 2 的 [hello-pass](../hello-pass/) 结构对应。
遍历每个 `func.func`，打印函数名和 Operation 数量。

## 快速开始

```bash
chmod +x run.sh
./run.sh
```

## 预期输出

```
Hello: add
  Operation 数量: 3
Hello: say_hello
  Operation 数量: 2
```

## 和 hello-pass 的对照

| | hello-pass (LLVM) | HelloMLIRPass (MLIR) |
|---|---|---|
| 遍历单位 | `Function &F` | `func::FuncOp` |
| 回调函数 | `run(Function &F)` | `runOnOperation()` |
| 打印函数名 | `F.getName()` | `func.getName()` |
| 统计子结构 | `BB.size()` (基本块) | `func.walk()` (Operation) |
| 运行方式 | `opt --passes="hello"` | 独立程序 `./hello-mlir` |

## 文件说明

| 文件 | 作用 |
|------|------|
| hello-mlir.cpp | 独立程序：解析 .mlir → 运行 Pass → 打印结果（45 行） |
| test.mlir | 测试输入（add + say_hello） |
| run.sh | 一键编译 + 运行 |
| CMakeLists.txt | 构建配置（备用） |

## 学完这个

- 深入理解 MLIR → [00-从LLVM到MLIR](../../docs/mlir/00-从LLVM到MLIR.md)
- 遇到术语 → [术语表](../../docs/glossary.md)
