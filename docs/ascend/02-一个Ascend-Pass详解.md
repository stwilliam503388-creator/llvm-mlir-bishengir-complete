# 02 — 一个 Ascend Pass 详解

> 目标：逐层拆解一个真实的 MLIR Pass
> 前置：[01 — Dialect 详解](./01-husion-hivm-Dialect详解.md)
> 预估时间：25 分钟

## 1. 选 ConvertLinalgToHusion

这是 Lowering 流程的第一个关键 Pass。和 mlir-hello 的思路一致：遍历 IR → 匹配 → 替换。

在 ascendnpu-ir 中：`bishengir/lib/Conversion/ConvertLinalgToHusion.cpp`

## 2. MLIR Pass 的三部分

| 部分 | 文件类型 | 内容 | mlir-hello 对应 |
|------|---------|------|---------------|
| ① Operation 定义 | `.td` (TableGen) | husion.elemwise_binary 长什么样 | test.mlir 里的 func.func |
| ② 匹配模式 | `.cpp` 中的 Pattern | 怎么找到逐元素运算 | func.walk() |
| ③ Pass 注册 | `.cpp` 中的 Plugin | 怎么加载到 mlir-opt | main() |

### ① Operation 定义 (.td)

```tablegen
def Husion_ElemwiseBinaryOp : Husion_Op<"elemwise_binary"> {
  let arguments = (ins
    StrAttr:$op_type,       // "add"/"mul"/"sub"
    AnyTensor:$inputs,
    AnyTensor:$outputs
  );
  let results = (outs AnyTensor:$result);
}
```

### ② 匹配模式（核心）

```cpp
struct LinalgGenericToElemwiseBinary : OpRewritePattern<linalg::GenericOp> {
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    // Step 1: 检查是不是逐元素运算
    if (!isElementwise(op)) return failure();

    // Step 2: 提取操作类型
    StringRef opType = getOpType(op);  // "add"

    // Step 3: 创建新的 husion 操作
    auto newOp = rewriter.create<HusionElemwiseBinaryOp>(
        op.getLoc(), opType, op.getInputs(), op.getOutputs());

    // Step 4: 替换
    rewriter.replaceOp(op, newOp);
    return success();
  }
};
```

**和 mlir-hello 的对比**：

| | mlir-hello | ConvertLinalgToHusion |
|---|---|---|
| 继承 | `PassWrapper<HelloMLIRPass, ...>` | `OpRewritePattern<GenericOp>` |
| 核心函数 | `runOnOperation()` | `matchAndRewrite(Operation, Rewriter)` |
| 遍历 | `func.walk()` | 框架自动 |
| 修改 IR | 不改 | `rewriter.replaceOp()` |

## 3. 看测试用例

**输入** (linalg, 15 行):
```mlir
func.func @simple_add(%A: tensor<128xf32>, %B: tensor<128xf32>)
    -> tensor<128xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<128xf32>, tensor<128xf32>)
    outs(%A : tensor<128xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<128xf32>
  func.return %0 : tensor<128xf32>
}
```

**输出** (husion, 3 行):
```mlir
func.func @simple_add(%A: tensor<128xf32>, %B: tensor<128xf32>)
    -> tensor<128xf32> {
  %0 = husion.elemwise_binary "add" ins(%A, %B) outs(%A)
      : tensor<128xf32>
  func.return %0 : tensor<128xf32>
}
```

**15 行 → 3 行**。Pass 做了什么一目了然。

## 4. 动手：加一个新功能

想支持 `arith.subf`（减法）：
1. 确认 `.td` 中 `elemwise_binary` 已支持 `"sub"`
2. `matchAndRewrite` 中加判断
3. 测试文件加一个减法用例

和 mlir-hello 挑战 2（统计 add 指令数）思路一样。

## 验证

- [ ] 能说出 MLIR Pass 的三部分
- [ ] 能对照测试用例说出 Pass 做了什么变换
- [ ] 能说出 `matchAndRewrite` 的 4 个步骤

> 📖 [术语表](../glossary.md)
> **下一步**：[03 — 构建与调试指南](./03-构建与调试指南.md)
