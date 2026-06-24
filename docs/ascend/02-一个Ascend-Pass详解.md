# 02 — 一个 Ascend Pass 详解

> 目标：理解真实 AscendNPU-IR Pass 的实现模式
> 前置：[01 — hivm 和 hacc Dialect 详解](./01-husion-hivm-Dialect详解.md)
> 预估时间：30 分钟

## 1. 从测试用例反推 Pass 做了什么

虽然源码被 git-lfs 保护，但测试用例清楚展示了每个 Pass 的效果。

**源文件位置**（在 AscendNPU-IR 中）：
- 测试：`bishengir/test/Integration/HIVM/VecAdd/add.mlir`
- 测试：`bishengir/test/Pass/pass-manager.mlir`
- 测试：`bishengir/test/Pass/cpu-runner.mlir`

## 2. ConvertLinalgToHivm — 最关键的 Pass

这个 Pass 把 MLIR 标准的 `linalg.generic` Lowering 为 Ascend 专用的 `hivm.hir.*` 指令。

### 输入 (linalg)

```mlir
func.func @add(%A: tensor<16xf16>, %B: tensor<16xf16>) -> tensor<16xf16> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<16xf16>, tensor<16xf16>)
    outs(%A : tensor<16xf16>) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %add = arith.addf %a, %b : f16
    linalg.yield %add : f16
  } -> tensor<16xf16>
  func.return %0 : tensor<16xf16>
}
```

### 输出 (hivm)

```mlir
func.func @add(%A: memref<16xf16, #hivm.address_space<gm>>,
               %B: memref<16xf16, #hivm.address_space<gm>>,
               %C: memref<16xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ub_a = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%A) outs(%ub_a)
  %ub_b = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%B) outs(%ub_b)
  %ub_c = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_c)
  hivm.hir.store ins(%ub_c) outs(%C)
  return
}
```

### 逐行对比 Pass 做了什么

| 输入 (linalg) | 输出 (hivm) | Pass 做了什么 |
|--------------|------------|--------------|
| `tensor<16xf16>` 参数 | `memref<..., gm>` 参数 | tensor→memref 转换，标记 Global Memory |
| （隐式） | `memref.alloc() : memref<..., ub>` | 分配 Unified Buffer |
| （隐式） | `hivm.hir.load ins(%A) outs(%ub_a)` | 显式加载：HBM→片上 |
| `linalg.generic { arith.addf }` | `hivm.hir.vadd ins(%ub_a, %ub_b) outs(%ub_c)` | 逐元素 add→向量 add |
| `func.return %0` | `hivm.hir.store ins(%ub_c) outs(%C)` | 显式存储：片上→HBM |
| （无） | `hacc.entry, #hacc.function_kind<DEVICE>` | 标记为 NPU kernel |

### Pass 的逻辑推断

```cpp
// ConvertLinalgToHivm 的核心逻辑（推断，实际代码在 bishengir/lib/Conversion/）
struct LinalgGenericToHivm : OpRewritePattern<linalg::GenericOp> {
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    // Step 1: 提取核心操作（addf, mulf, subf 等）
    Operation *innerOp = getInnerOp(op);  // arith.addf

    // Step 2: 分配 Unified Buffer
    auto ubType = MemRefType::get(shape, elemType, ubSpace);
    Value ub_a = rewriter.create<memref::AllocOp>(loc, ubType);

    // Step 3: 创建 hivm.hir.load
    rewriter.create<hivm::hir::LoadOp>(loc, ub_a, gm_arg);

    // Step 4: 创建 hivm.hir.vadd（根据 innerOp 类型）
    if (isa<arith::AddFOp>(innerOp))
      rewriter.create<hivm::hir::VAddOp>(loc, ub_a, ub_b, ub_c);

    // Step 5: 创建 hivm.hir.store
    rewriter.create<hivm::hir::StoreOp>(loc, ub_c, gm_out);

    // Step 6: 添加 kernel 属性
    func->setAttr("hacc.entry", UnitAttr::get(ctx));
    func->setAttr("hacc.function_kind",
                  hacc::FunctionKindAttr::get(ctx, hacc::FunctionKind::DEVICE));

    rewriter.eraseOp(op);  // 删除原来的 linalg.generic
    return success();
  }
};
```

**和 mlir-hello 的对比**：

| | mlir-hello | ConvertLinalgToHivm |
|---|---|---|
| 遍历 | `func.walk()` | `OpRewritePattern` |
| 做判断 | （不判断） | `getInnerOp()` 识别 add/sub/mul |
| 创建新操作 | （不创建） | `create<LoadOp>`, `create<VAddOp>`, `create<StoreOp>` |
| 修改 IR | 不改 | 替换整个函数体 |
| 行数 | 45 行 | ~200 行 |

## 3. 源码导读（文件位置）

| 想看什么 | 文件（在 AscendNPU-IR 中） |
|---------|--------------------------|
| hivm 操作定义 | `bishengir/include/.../HIVM/` |
| LoadOp / StoreOp | 同上，`.td` TableGen 文件 |
| 地址空间定义 | `#hivm.address_space` 属性 |
| 转换 Pass | `bishengir/lib/Conversion/` |
| 融合优化 | `bishengir/lib/Transforms/` |
| 测试用例 | `bishengir/test/Integration/HIVM/` |

## 4. 验证所学

去 AscendNPU-IR 里找一个测试文件，画出 IR 变换前后对照：
```bash
cd AscendNPU-IR
cat bishengir/test/Integration/HIVM/VecAdd/add.mlir
# 这个文件本身就是 hivm IR
# 和 ascend-samples/03 的 input.mlir 对比，看 linalg→hivm 的差异
```

> 📖 [术语表](../glossary.md)
> **下一步**：[03 — 构建与调试指南](./03-构建与调试指南.md)
