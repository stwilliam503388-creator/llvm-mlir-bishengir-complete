---
created: 2026-06-21
tags: [mlir, pass, AscendNPU-IR, pattern-rewriting]
aliases: [自定义 MLIR Pass, AscendNPU-IR Pass 实战]
---

# 自定义 AscendNPU-IR MLIR Pass 实战

> 基于 Toy Tutorial 学到的知识，写两个针对 AscendNPU-IR 的自定义 Pass。
> 源码位置：`~/hermes-workspace/toy-tutorial/ascendnpu-ir-op-counter/`

---

## 一、两个 Pass 概览

| Pass | 类型 | 用途 | 对应 Toy Tutorial |
|------|------|------|------------------|
| **OpCounterPass** | 分析 Pass | 统计 module 中各 op 出现次数 | Ch3 ShapeInferencePass |
| **PeelTransposePass** | 转换 Pass | 消除冗余 transpose 和融合二元操作 | Ch3 ToyCombine + Ch5 Lowering |

---

## 二、OpCounterPass（分析 Pass）

### 核心代码

```cpp
// OperationPass<ModuleOp> — 作用于整个 module
class OpCounterPass
    : public PassWrapper<OpCounterPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "ascendnpu-ir-op-counter"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::map<std::string, int> opCount;

    // walk() 遍历所有操作（嵌套 Region 内也遍历）
    module->walk([&](Operation *op) {
      std::string opName = op->getName().getStringRef().str();
      opCount[opName]++;
    });

    // 输出统计
    for (const auto &[name, count] : opCount)
      llvm::outs() << "  " << name << ": " << count << "\n";
  }
};
```

### walk() API 详解

MLIR 的遍历机制有多种层级，用于不同场景：

| 方式 | 用法 | 作用范围 | 适用场景 |
|------|------|---------|---------|
| **walk()** | `module->walk([](Operation *op) {...})` | 全部 ops（递归） | 分析 Pass |
| **walk\<T\>()** | `module->walk([](arith::AddFOp op) {...})` | 只匹配 T 类型 ops | 特定 op 分析 |
| **getOps\<T\>()** | `func.getOps<func::FuncOp>()` | 直接子级 ops | 遍历函数 |
| **OpIterator** | `func->getRegions()` → block → ops | 手动控制 | 复杂遍历 |
| **Pattern** | `matchAndRewrite()` | 匹配+重写自动 | 转换 Pass |

### walk() 源码追踪（AscendNPU-IR 实际用例）

```
LinalgToHFusion.cpp 中的匿名 walk：
  getOperation()->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::GenericOp>(op)) {
      // 处理每个 linalg.generic
    }
  });
```

---

## 三、PeelTransposePass（转换 Pass）

### Pattern 1：冗余 add 转 mul

```
匹配:
  %1 = hfusion.elemwise_binary {fun = add}(%A, %B)
  %2 = hfusion.elemwise_binary {fun = add}(%1, %1)   ← x + x

重写:
  %1 = hfusion.elemwise_binary {fun = add}(%A, %B)
  %2 = hfusion.elemwise_binary {fun = mul}(%1, 2.0)   ← x * 2
```

### 源码结构

```cpp
struct RedundantAddToMul : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
  LogicalResult matchAndRewrite(
      hfusion::ElemwiseBinaryOp op,
      PatternRewriter &rewriter) const override {

    // Step 1: 检查 fun 是否为 add
    if (op.getFun() != hfusion::BinaryFn::add)
      return failure();

    // Step 2: 检查 lhs 和 rhs 是否相同
    if (op.getLhs() != op.getRhs())
      return failure();

    // Step 3: 检查 lhs 是否来自另一个 elemwise_binary
    auto parent = op.getLhs().getDefiningOp<hfusion::ElemwiseBinaryOp>();
    if (!parent || parent.getFun() != hfusion::BinaryFn::add)
      return failure();

    // Step 4: 重写
    auto two = rewriter.create<hfusion::ConstantOp>(loc, 2.0);
    auto mul = rewriter.create<hfusion::ElemwiseBinaryOp>(
        loc, hfusion::BinaryFn::mul, parent.getResult(), two);
    rewriter.replaceOp(op, mul);

    return success();
  }
};
```

### 注册 Pass

```cpp
// 与 AscendNPU-IR 现有 Pass 完全相同的注册方式
void registerPeelTransposePass() {
  PassRegistration<PeleTransposePass> reg;
}

// 在 InitAllPasses.h 中添加：
//   void registerPeelTransposePass();
```

---

## 四、MLIR Pass 两种模式对照

### 4.1 分析 Pass

```cpp
class AnalysisPass : public PassWrapper<AP, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 1. 读取 IR：walk() / getOps<>()
    // 2. 分析，不修改
    // 3. 输出结果
    // 不调用 signalPassFailure()
  }
};
```

### 4.2 转换 Pass（Pattern 模式）

```cpp
class TransformPass : public PassWrapper<TP, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MyPattern>(&getContext());
    // 贪婪应用直到收敛
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

### 4.3 转换 Pass（Dialect Conversion 模式）

```cpp
// 这是 AscendNPU-IR 现有 Pass 用的模式
class ConversionPass : public PassWrapper<CP, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<TargetDialect>();    // 允许目标 dialect
    target.addIllegalDialect<SourceDialect>();  // 禁止源 dialect

    RewritePatternSet patterns(&getContext());
    patterns.add<SourceToTargetPattern>(&getContext());
    applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};
```

---

## 五、三种 Pass 模式在 AscendNPU-IR 中的实际分布

| 模式 | AscendNPU-IR 实际 Pass | Toy Tutorial 对照 |
|------|-------------------|------------------|
| **Analysis（walk）** | 无（AscendNPU-IR 目前无分析 Pass） | ShapeInferencePass |
| **Pattern（greedy）** | （可添加） | ToyCombine |
| **Conversion** | **ConvertLinalgToHFusion** | LowerToAffineLoops |
| **Conversion** | **ConvertHFusionToHIVM** | LowerToAffineLoops |

### bishengir-opt 命令行（如果有的话）

```bash
bishengir-opt \
  -convert-linalg-to-hfusion \    # Dialect Conversion
  -convert-arith-to-hfusion \     # Dialect Conversion  
  -convert-hfusion-to-hivm \      # Dialect Conversion
  -ascendnpu-ir-op-counter \         # ★ 自定义分析 Pass
  -bishengir-peel-transpose \     # ★ 自定义转换 Pass
  vecadd.mlir
```

---

## 六、与 Toy Tutorial 的关键技术对照

| 技术点 | Toy Tutorial | AscendNPU-IR | 自定义 Pass |
|--------|-------------|-----------|------------|
| **Op 类型匹配** | `toy::AddOp` | `hfusion::ElemwiseBinaryOp` | `OpRewritePattern<T>` |
| **分析遍历** | `func->walk([](Op*){})` | 同上 | 同上 |
| **创建 Op** | `rewriter.create<toy::AddOp>()` | `rewriter.create<hfusion::...>()` | `rewriter.create<>()` |
| **替换 Op** | `rewriter.replaceOp(old, new)` | 同上 | 同上 |
| **删除 Op** | `rewriter.eraseOp(op)` | 同上 | 同上 |
| **Pattern 注册** | `patterns.add<AddOpLowering>()` | `patterns.add<Convert...>()` | `patterns.add<MyPattern>()` |
| **Pass 注册** | `PassRegistration<Pass>` | 同上 | 同上 |
| **命令行参数** | `getArgument()` 返回名字 | 同上 | 同上 |

---

## 七、文件位置

```text
~/hermes-workspace/toy-tutorial/
├── example.toy                              # Toy 语言测试文件
├── ascendnpu-ir-op-counter/
│   ├── BishengirOpCounter.cpp               # ★ 分析 Pass（op 计数）
│   └── BishengirPeelTranspose.cpp           # ★ 转换 Pass（冗余消除）
├── src/Ch1~Ch7/                              # Toy Tutorial 官方源码
└── README.md                                 # 工作区说明
```
