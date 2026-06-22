# 03：动手——看 MLIR 长什么样

> 阅读时间：10 分钟 | 前置知识：Primer 00, 01, 02
> 需要：Homebrew LLVM `mlir-opt` 可用

---

这是最重要的 Primer 章节——不再读概念，而是**亲手运行命令看 IR 变化**。

---

## 3.1 看一下 MLIR 长什么样

打开项目中的一个 MLIR 文件：

```bash
head -30 projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir
```

你会看到：

```mlir
module {
  func.func @vecadd(%A: memref<128xf16>, %B: memref<128xf16>, %C: memref<128xf16>) {
    linalg.generic {...} ins(%A, %B : ...) outs(%C : ...) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    }
    return
  }
}
```

不用看懂所有语法。只要注意一件事：**这看起来不像任何编程语言**。它既不像 Python，也不像 C++，也不像汇编。这就是 IR——编译器的内部表示。

关键模式：`linalg.generic { 里面写运算是做什么的 }` —— "我要做逐元素加法"。

---

## 3.2 降级一步看一步

现在做 lowering，看看 IR 是怎么一步步变啰嗦的：

```bash
# 导出路径（仅限 Homebrew LLVM）
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Step 1: 只看原始 IR（什么都不做）
mlir-opt projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir
```

输出和上面一样，只是帮你验证 `mlir-opt` 能工作。

```bash
# Step 2: 降一级 → linalg → affine
mlir-opt --convert-linalg-to-affine-loops \
  projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir 2>&1
```

注意输出里的 `affine.for`——从"我要做加法"变成了"循环 0 到 127，每次取一个元素做加法"。

```bash
# Step 3: 降到 LLVM IR
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
  --convert-scf-to-cf --convert-func-to-llvm \
  projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir 2>&1
```

输出变成了 `%1 = load`、`%2 = add`、`%3 = store`——真正的机器指令。从 3 行 MLIR 变成了约 38 行 LLVM。

---

## 3.3 对比不同操作的膨胀率

```bash
# 看一个简单的操作（ReLU）
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
  --convert-scf-to-cf --convert-func-to-llvm \
  projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/02_relu.mlir 2>&1 | wc -l
# 输出约 42 行

# 看一个复杂的操作（矩阵乘法）
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
  --convert-scf-to-cf --convert-func-to-llvm \
  projects/ascendnpu-ir-demo/test-cases/mlir/03_advanced/01_matmul.mlir 2>&1 | wc -l
# 输出约 74 行
```

| 操作 | MLIR 输入（行） | LLVM 输出（行） | 膨胀率 |
|------|---------------|---------------|--------|
| VecAdd | 3 | 38 | 12.7× |
| ReLU | 5 | 42 | 8.4× |
| **MatMul** | **1** | **74** | **74×** |

这就是"高级 IR 保留语义"的价值——bishengir 把 `linalg.matmul` 保留为 1 行 `hivm.mmul` NPU 指令，而不是展开成 74 行标量运算。

---

## 3.4 看 Triton 代码的对应关系

现在打开 Triton 代码，和 MLIR 对比：

```bash
head -20 projects/ascendnpu-ir-demo/test-cases/triton/01_basic/01_vecadd.py
```

```python
# (对应 mlir/01_basic/01_vecadd.mlir)
@triton.jit
def vecadd_kernel(A, B, C, N, BLOCK):
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    c = a + b         # ← 这就是 linalg.generic + arith.addf
    tl.store(C + offsets, c)
```

注意：Triton 的 `a + b` 一行，对应 MLIR 中多行 `linalg.generic` + `arith.addf`，再对应 LLVM 中 38 行指令。

**完整映射表**在 `test-cases/triton/MAPPING.md` 和 `test-cases/mlir/MAPPING.md`。

> ✅ **检查自己**：
> 1. 运行 `mlir-opt` 看一次 vecadd 降级，从 MLIR 到 LLVM 发生了什么变化？
> 2. 矩阵乘法（matmul）的膨胀率为什么这么大？
>    → 因为矩阵乘法需要三重循环展开成标量运算。bishengir 保留 1 行 NPU 指令避免这个膨胀。
