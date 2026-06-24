# 00 — 从 LLVM 到 MLIR

> 目标：用你已经学会的 LLVM 概念来理解 MLIR
> 前置：[Phase 2 LLVM](../llvm/README.md)
> 预估时间：15 分钟

## 1. MLIR 解决什么问题？

你已经学会 LLVM 了——写 Pass、读 IR、跑 opt。那为什么还需要 MLIR？

**LLVM 的问题：只有一层 IR。**

```
C 源码 → LLVM IR → 优化 → LLVM IR → 汇编
              ↑_____________↑
              所有优化都在同一层
```

不管做什么优化——死代码消除、循环展开、向量化——都在同一层 LLVM IR 上操作。就像**一个建筑工地上，木工、电工、水管工都在同一个房间干活**，互相踩脚。

**MLIR 的方案：多层 IR，每层做最适合的事。**

```
Tensor IR (高级)  →  循环 IR (中级)  →  向量 IR (低级)  →  LLVM IR
    ↑                    ↑                ↑               ↑
  tensor dialect      scf dialect     vector dialect    llvm dialect
  描述矩阵乘法        描述循环结构      描述向量指令      最终生成代码
```

> **一句话**：LLVM = 通用语言（所有人都说英语）；MLIR = 专用语言系统（数学家说数学符号，电工说电路图）。

## 2. 关键概念对照

| LLVM 概念 | MLIR 对应 | 区别 |
|-----------|----------|------|
| `Function` | `func.func` Operation | MLIR 中函数也是一个 Operation |
| `BasicBlock` | `Block` | 基本相同 |
| `Instruction` | `Operation` | MLIR 更通用，可多返回值 |
| `Module` | `ModuleOp` | 相同 |
| `Pass` (`FunctionPass`) | `Pass` (`mlir::Pass`) | 接口不同，理念相同 |
| `.ll` 文件 | `.mlir` 文件 | 语法不同，都有 SSA |
| `i32` / `float` | `i32` / `f32` | 兼容 |
| `%variable` | `%variable` | SSA 变量命名相同 |

## 3. Dialect：MLIR 的核心创新

**Dialect（方言）= 一组自定义的 Operation + Type。**

| Dialect | 用途 | 类比 |
|---------|------|------|
| `arith` | 整数/浮点算术 | 计算器 |
| `scf` | 控制流（for/while/if） | 流程图 |
| `linalg` | 线性代数（矩阵乘） | 数学公式 |
| `func` | 函数定义和调用 | C 的函数 |
| `memref` | 内存访问 | 货架 |
| `llvm` | LLVM IR 直接映射 | 出口 |

### 一个 .mlir 文件的解剖

```mlir
func.func @max(%a: i32, %b: i32) -> i32 {
  %cmp = arith.cmpi sgt, %a, %b : i32          // arith dialect
  %result = scf.if %cmp -> i32 {                // scf dialect
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  func.return %result : i32                     // func dialect
}
```

同一个文件里三个 Dialect 混用——MLIR 的灵活性就在这里。

## 4. 从 HelloPass 到 MLIR Pass

| | HelloPass (LLVM) | mlir-hello (MLIR) |
|---|---|---|
| 遍历单位 | `Function &F` | `func::FuncOp` |
| 回调函数 | `run(Function &F)` | `runOnOperation()` |
| 打印 | `F.getName()` | `func.getName()` |
| 统计 | `BB.size()` | `func.walk()` |

**你已经会写 LLVM Pass，MLIR Pass 只是换了 API。**

## 验证

- [ ] 能说出 MLIR 比 LLVM 多了什么（多层 IR + Dialect）
- [ ] 能指出 `.mlir` 中 `arith.addi` 和 `scf.if` 的 Dialect
- [ ] 能说出 MLIR Pass 和 LLVM Pass 的相同点

> 📖 [术语表](../glossary.md)
> **下一步**：[01 — Toy Tutorial 导读](./01-Toy-Tutorial导读.md)
