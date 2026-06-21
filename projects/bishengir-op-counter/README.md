# bishengir-op-counter — 自定义 MLIR Pass 参考代码

针对 bishengir (ascendnpu-ir) dialect 编写的两个自定义 Pass，演示 MLIR 的两种核心模式。

## 结构

```
bishengir-op-counter/
├── BishengirOpCounter.cpp       — 分析 Pass
└── BishengirPeelTranspose.cpp   — 转换 Pass
```

## Pass 说明

### 1. OpCounter — 操作统计

```
bishengir-opt vecadd.mlir --count-ops
→
Statistics:
  func.func:   1
  linalg.generic: 1
  arith.addf:  1
  ...
```

**模式**: `Pass::runOnOperation()` 中调用 `op->walk()` 遍历所有嵌套操作
**对照**: Toy Tutorial Ch3 `ShapeInferencePass`

### 2. PeelTranspose — 冗余消除

检测 `add(transpose(A), transpose(B))` 模式，将其转换为 `transpose(add(A, B))`，提前转置融合。

**模式**: `OpRewritePattern` + `ConversionTarget` + `applyPartialConversion`
**对照**: Toy Tutorial Ch3 `ToyCombine.cpp` (transpose folding)

## 编译

需要 bishengir 的构建系统。在 bishengir 源码树中:

```cmake
# 添加到 bishengir/lib/Conversion/CMakeLists.txt
add_mlir_conversion_library(BishengirOpCounter
    BishengirOpCounter.cpp
    DEPENDS
    MLIRBishengirOpsIncGen
)
```

或作为 standalone Pass（参考 `standalone-mlir` 项目中的 `standalone-opt.cpp`）：

```cpp
int main() {
    registerPasses();
    return MlirOptMain(...);
}
```

## 关键模式

```cpp
// 分析 Pass: walk + 统计
void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
        counters[op->getName().getStringRef()]++;
    });
}

// 转换 Pass: Pattern + ConversionTarget
LogicalResult matchAndRewrite(TransposeOp op, ...) {
    if (matchRedundantTranspose(op.getOperand()))
        return replaceOpWithNewOp<TransposeOp>(...);
    return failure();
}
```
