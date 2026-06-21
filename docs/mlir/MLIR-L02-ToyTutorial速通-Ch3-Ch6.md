---

created: 2026-06-21
tags: [mlir, toy-tutorial, pattern-matching, lowering]
aliases: [Toy Tutorial Ch3-6]

  - 工程
---

# Toy Tutorial Ch3~6：优化 + 降级 + 代码生成

## 五、Ch3：模式匹配优化（Shape Inference + 常量折叠）

### 5.1 架构变化

```
Ch2:  .toy → Toy Dialect (raw IR)
Ch3:  .toy → Toy Dialect → ShapeInference → Canonicalize → 优化的 Toy Dialect
                                          ↓
                                    + ToyCombine 模式匹配
```

### 5.2 ShapeInferencePass

Toy operations **初始没有形状信息**（张量维度是未知的）：

```mlir
// 形状推断前：tensor<*xf64>（* 表示未知）
%0 = toy.constant { dense<[[1,2],[3,4]]> } : tensor<*xf64>
%1 = toy.transpose(%0) : tensor<*xf64> -> tensor<*xf64>

// 形状推断后：tensor<2x2xf64>
%0 = toy.constant { dense<[[1,2],[3,4]]> } : tensor<2x2xf64>
%1 = toy.transpose(%0) : tensor<2x2xf64> -> tensor<2x2xf64>
```

**ShapeInferencePass 源码**（`ShapeInferencePass.cpp`）：

```cpp
// 这是一个 OperationPass，但不是作用于 func.func，
// 而是作用于 toy 模块内所有操作
struct ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {

  void runOnOperation() override {
    auto func = getOperation();

    // 迭代固定点算法：不断传播形状，直到收敛
    while (changed) {
      changed = false;
      func->walk([&](Operation *op) {
        // 对每个操作，检查是否需要推断形状
        if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
          // ShapeInference 接口定义在 Ops.td 中：
          // def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
          //   let methods = [... inferShapes() ...]
          // }
          shapeOp.inferShapes();
          changed = true;
        }
      });
    }
  }
};
```

**关键点**：
- 使用 **fixed-point iteration**（固定点迭代）——不断推断直到形状不再变化
- 依赖 `ShapeInference` 接口（OpInterface）——每个 op 自己实现 `inferShapes()`
- 是 **Interprocedural**（跨过程）的——函数调用也会传递形状

### 5.3 Pattern Rewriting（优化模式）

**写在 `ToyCombine.td` 中的优化规则**：

```tablegen
// 模式 1: reshape(reshape(x)) → reshape(x)
// 连续两次 reshape 是冗余的
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp (ReshapeOp $arg)),
  (ReshapeOp $arg)>;

// 模式 2: transpose(transpose(x)) → x
// 两次转置抵消
def TransposeTransposeOptPattern : Pat<
  (TransposeOp (TransposeOp $arg)),
  (replaceWithValue $arg)>;

// 模式 3: reshape(transpose(reshape(x))) → transpose(x)
// 不必要的 reshape 可以消除
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp (TransposeOp (ReshapeOp $arg))),
  (TransposeOp $arg)>;
```

**手动写 C++ Pattern**（`ToyCombine.cpp`）：

```cpp
// 在 Ch5 中，用 C++ 写的融合模式：
// transpose(x) * transpose(y) → multiply_transpose(x, y)
struct FuseTransposeMul : public OpRewritePattern<MulOp> {
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // 检查 mul 的两个输入是否都是 transpose
    auto lhsTranspose = op.getLhs().getDefiningOp<TransposeOp>();
    auto rhsTranspose = op.getRhs().getDefiningOp<TransposeOp>();

    if (!lhsTranspose || !rhsTranspose)
      return failure();  // 不匹配

    // 重写：用 TransposeOp(input_a * input_b) 代替
    Value newMul = rewriter.create<MulOp>(op.getLoc(),
                     lhsTranspose.getInput(), rhsTranspose.getInput());
    rewriter.replaceOpWithNewOp<TransposeOp>(op, newMul);
    return success();
  }
};
```

### 5.4 Canonicalize Pass

MLIR 内置的 canonicalization pass 会自动调用 op 的 `getCanonicalizationPatterns()`：

```cpp
// 在 Ops.td 中声明
def TransposeOp : Toy_Op<"transpose"> {
  let hasCanonicalizeMethod = 1;  // 启用 canonicalization
}

// 实现
mlir::LogicalResult TransposeOp::canonicalize(
    TransposeOp op, PatternRewriter &rewriter) {
  if (auto parent = op.getInput().getDefiningOp<TransposeOp>()) {
    // transpose(transpose(x)) → x
    rewriter.replaceOp(op, parent.getInput());
    return success();
  }
  return failure();
}
```

---

## 六、Ch5：Partial Lowering（局部降级）

### 6.1 什么是 Partial Lowering

把 Toy dialect 的**部分操作**降低到 MLIR 标准 dialect（affine、scf、arith、memref），但保留另一些高层操作。

```
Toy Dialect                       标准 MLIR Dialect
─────────────────                 ─────────────────
toy.constant      ───────→        arith.constant
toy.add           ───────→        linalg.generic 或 affine.for
toy.mul           ───────→        linalg.generic 或 affine.for
toy.transpose     ───────→        affine.for (重新排列索引)
toy.print         ───────→        func.call @print (运行时)
toy.return        ───────→        func.return
```

### 6.2 `LowerToAffineLoops` Pass

**文件**：`Ch5/mlir/LowerToAffineLoops.cpp`

```cpp
// 这是 AscendNPU-IR 的 ConvertLinalgToHFusion / ConvertHFusionToHIVM 的 Toy 版对照

void ToyToAffineLoweringPass::runOnOperation() {
  // Step 1: 设置 ConversionTarget（哪些算"合法"）
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<affine::AffineDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<scf::SCFDialect>();

  // Toy dialect 的操作只有在被显式处理后才是合法的
  target.addIllegalOp<AddOp, MulOp, TransposeOp, ConstantOp, ReturnOp>();

  // Step 2: 注册转换规则
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, MulOpLowering, TransposeOpLowering,
               ConstantOpLowering, ReturnOpLowering>(&getContext());

  // Step 3: 执行
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

### 6.3 逐操作降级实例

**`toy.add` 降低为 `affine.for` 循环**：

```cpp
struct AddOpLowering : public OpRewritePattern<toy::AddOp> {
  LogicalResult matchAndRewrite(toy::AddOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 分配输出内存
    auto memRefType = getMemRefType(op.getType());
    auto alloc = rewriter.create<memref::AllocOp>(op.getLoc(), memRefType);

    // 2. 创建 affine.for 循环遍历所有元素
    auto loop = rewriter.create<affine::AffineForOp>(op.getLoc(), 0, size);
    auto iv = loop.getInductionVar();

    // 3. 加载 lhs[i], rhs[i]
    auto loadedLhs = rewriter.create<affine::AffineLoadOp>(loc, lhs, iv);
    auto loadedRhs = rewriter.create<affine::AffineLoadOp>(loc, rhs, iv);

    // 4. 执行加法
    auto add = rewriter.create<arith::AddFOp>(loc, loadedLhs, loadedRhs);

    // 5. 存储到 output[i]
    rewriter.create<affine::AffineStoreOp>(loc, add, alloc, iv);

    // 6. 替换原 op 的输出
    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

**降低后的 IR**：

```mlir
// 降低前：toy dialect
%2 = toy.add %0, %1 : tensor<4x4xf64>

// 降低后：affine + arith + memref
%alloc = memref.alloc() : memref<4x4xf64>
scf.for %i = 0 to 4 {
  scf.for %j = 0 to 4 {
    %a = affine.load %0[%i, %j] : memref<4x4xf64>
    %b = affine.load %1[%i, %j] : memref<4x4xf64>
    %sum = arith.addf %a, %b : f64
    affine.store %sum, %alloc[%i, %j] : memref<4x4xf64>
  }
}
```

### 6.4 Toy → AscendNPU-IR 对照

| 概念 | Toy Tutorial | AscendNPU-IR |
|------|-------------|-----------|
| **高级 dialect** | `toy.constant/add/mul/transpose` | `hfusion.elemwise_binary{add}` |
| **低级 dialect** | `affine.for + arith.addf + memref` | `hivm.vadd + hivm.load + hivm.store` |
| **Pass 名字** | `-lower-toy-to-affine` | `-convert-linalg-to-hfusion` |
| **转换方式** | `applyPartialConversion` | `applyPartialConversion` |
| **ConversionTarget** | 禁止 Toy Ops，允许 Affine/Arith/MemRef | 禁止 Linalg Ops，允许 HFusion Ops |

**核心模式完全一致**：
```
ConversionTarget.setup()      → 定义"合法" vs "非法" dialect
RewritePatternSet.add()        → 注册每条转换规则
applyPartialConversion()       → 一次性批量转换
```

---

## 七、Ch6：Lowering to LLVM（代码生成）

### 7.1 完整流水线

```
.toy 源文件
  │
  ├── toyc -emit=mlir          → Toy Dialect IR
  ├── toyc -emit=mlir-affine   → Affine + MemRef IR
  ├── toyc -emit=mlir-llvm     → LLVM Dialect IR
  └── toyc -emit=llvm          → LLVM IR（可 JIT 执行）
```

### 7.2 LowerToAffineLoops + LowerToLLVM

Ch6 把剩余的操作也降低到 LLVM dialect：

```cpp
// LowerToLLVM 把以下操作降低：
// toy.print      → func.call @print  (调用运行时函数)
// func.func      → llvm.func
// memref.alloc   → llvm.alloca + llvm.mlir.addressof
// affine.load    → llvm.load
// affine.store   → llvm.store

struct PrintOpLowering : public OpRewritePattern<toy::PrintOp> {
  LogicalResult matchAndRewrite(toy::PrintOp op,
                                PatternRewriter &rewriter) const override {
    // 调用 C 运行时库的 print 函数
    auto module = op->getParentOfType<ModuleOp>();
    auto printFunc = module.lookupSymbol<LLVMFuncOp>("print_memref");

    rewriter.create<func::CallOp>(loc, printFunc, op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};
```

### 7.3 完整流水线实战

用 `mlir-opt` 可以模拟 Toy → LLVM 的效果（虽然 Toy dialect 编译不了，但标准 MLIR 可以）：

```bash
export LLVM_DIR="/opt/homebrew/opt/llvm"
export PATH="$LLVM_DIR/bin:$PATH"

# 标准 MLIR 降级流水线（与 Toy 的 LowerToLLVM 含义相同）
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  vecadd.mlir
```

### 7.4 Toy Tutorial 完整编译流水线（如果有 LLVM 源码编译）

```bash
# 只在完整 LLVM 源码编译下可用
./bin/toyc example.toy -emit=mlir           # → Toy Dialect IR
./bin/toyc example.toy -emit=mlir-affine    # → Affine + MemRef IR
./bin/toyc example.toy -emit=mlir-llvm      # → LLVM Dialect IR
./bin/toyc example.toy -emit=llvm            # → LLVM IR（可 JIT）
```

---

## 八、七章总结

| Chapter | 核心内容 | 关键概念 | 与 AscendNPU-IR 对照 |
|---------|---------|----------|------------------|
| **Ch1** | Toy AST/Parser | Lexer, Parser, AST | 前端解析器（不同项目各有差异）|
| **Ch2 ⭐** | **Toy Dialect 定义** | **TableGen Ops.td, MLIRGen** | **HFusion/HIVM .td 定义** |
| **Ch3** | Shape Inference | OpInterface, Fixed-point iteration | 类似的接口系统 |
| **Ch4** | Generic Shape Inference | 更通用的 ShapeInferencePatterns | 扩展到自定义接口 |
| **Ch5 ⭐** | **Partial Lowering** | **ConversionTarget, RewritePattern** | **Linalg→HFusion→HIVM 转换** |
| **Ch6** | Lower to LLVM | CodeGen, Runtime functions | → 最终目标指令 |
| **Ch7** | Debug Info | MLIR 的 debug 信息传递 | 可选 |
