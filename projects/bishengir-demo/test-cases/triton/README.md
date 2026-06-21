1|# Triton 代码示例
2|
3|> 28 个 MLIR 测试用例反向生成的 Triton Python 代码。
4|> 每文件对应 `test-cases/` 下同编号的 MLIR 文件。
5|
6|## 完整映射
7|
8|见 `MAPPING.md` — 包含：每文件的 Triton 代码 ↔ MLIR 代码 ↔ 公式 ↔ AI 角色 ↔ 降级模式对照。
9|
10|| 级别 | MLIR 路径 | Triton 文件 | 操作 |
11||------|----------|------------|------|
12|| ⭐ basic | `mlir/01_basic/01_vecadd.mlir` | `01_vecadd.py` | 向量加法 (残差连接) |
13|| ⭐ basic | `mlir/01_basic/02_relu.mlir` | `02_relu.py` | ReLU 激活 |
14|| ⭐ basic | `mlir/01_basic/03_tanh.mlir` | `03_tanh.py` | Tanh 激活 |
15|| ⭐ basic | `mlir/01_basic/04_softmax_exp.mlir` | `04_softmax_exp.py` | Softmax 指数 |
16|| ⭐ basic | `mlir/01_basic/05_broadcast.mlir` | `05_broadcast.py` | 标量广播 |
17|| ⭐ basic | `mlir/01_basic/06_dropout.mlir` | `06_dropout.py` | Dropout |
18|| ⭐ basic | `mlir/01_basic/07_fill.mlir` | `07_fill.py` | 常量填充 |
19|| ⭐ basic | `mlir/01_basic/08_fused.mlir` | `08_fused.py` | 算子融合 |
20|| ⭐⭐ intermediate | `mlir/02_intermediate/01_sigmoid.mlir` | `09_sigmoid.py` | Sigmoid |
21|| ⭐⭐ intermediate | `mlir/02_intermediate/02_silu.mlir` | `10_silu.py` | SiLU (LLaMA) |
22|| ⭐⭐ intermediate | `mlir/02_intermediate/03_leaky_relu.mlir` | `11_leaky_relu.py` | LeakyReLU |
23|| ⭐⭐ intermediate | `mlir/02_intermediate/04_gelu_tanh.mlir` | `12_gelu_tanh.py` | GELU (BERT) |
24|| ⭐⭐ intermediate | `mlir/02_intermediate/05_hard_sigmoid.mlir` | `13_hard_sigmoid.py` | Hard Sigmoid |
25|| ⭐⭐ intermediate | `mlir/02_intermediate/06_prelu.mlir` | `14_prelu.py` | PReLU |
26|| ⭐⭐ intermediate | `mlir/02_intermediate/07_reduce_sum.mlir` | `15_reduce_sum.py` | 归约求和 |
27|| ⭐⭐ intermediate | `mlir/02_intermediate/08_reduce_max.mlir` | `16_reduce_max.py` | 归约最大值 |
28|| ⭐⭐ intermediate | `mlir/02_intermediate/09_softmax_complete.mlir` | `17_softmax_complete.py` | Softmax 稳定 |
29|| ⭐⭐ intermediate | `mlir/02_intermediate/10_clamp.mlir` | `18_clamp.py` | 数值裁剪 |
30|| ⭐⭐ intermediate | `mlir/02_intermediate/11_layer_norm.mlir` | `19_layer_norm.py` | LayerNorm |
31|| ⭐⭐⭐ advanced | `mlir/03_advanced/01_matmul.mlir` | `20_matmul.py` | 矩阵乘法 |
32|| ⭐⭐⭐ advanced | `mlir/03_advanced/02_gemm_relu.mlir` | `21_gemm_relu.py` | GEMM+ReLU |
33|| ⭐⭐⭐ advanced | `mlir/03_advanced/04_conv2d.mlir` | `22_conv2d.py` | 二维卷积 |
34|| ⭐⭐⭐ advanced | `mlir/03_advanced/05_max_pool.mlir` | `23_max_pool.py` | 最大池化 |
35|| ⭐⭐⭐ advanced | `mlir/03_advanced/06_avg_pool.mlir` | `24_avg_pool.py` | 平均池化 |
36|| ⭐⭐⭐ advanced | `mlir/03_advanced/07_global_avg_pool.mlir` | `25_global_avg_pool.py` | 全局平均池化 |
37|| ⭐⭐⭐ advanced | `mlir/03_advanced/03_depthwise_conv.mlir` | `26_depthwise_conv.py` | 深度卷积 |
38|| ⭐⭐⭐ advanced | `mlir/03_advanced/08_batch_norm_part1.mlir` | `27_batch_norm_part1.py` | BN 均值 |
39|| ⭐⭐⭐ advanced | `mlir/03_advanced/09_batch_norm_part2.mlir` | `28_batch_norm_part2.py` | BN 标准化 |
40|
41|## 运行
42|
43|需 NVIDIA GPU + CUDA (Triton 无法在 Apple Silicon 上运行):
44|
45|```bash
46|cd projects/bishengir-demo/
47|python3 test-cases/triton/20_matmul.py   # 矩阵乘
48|python3 test-cases/triton/01_vecadd.py    # 向量加
49|python3 test-cases/triton/10_silu.py      # SiLU
50|```
51|
## MLIR → Triton 对照速查

| MLIR 模式 | Triton 等价 | 说明 |
|-----------|------------|------|
| `linalg.generic + arith.addf` | `a + b` | 逐元素运算法 |
| `arith.cmpf + arith.select` | `tl.where(cond, a, b)` | 条件分支 |
| `math.exp / math.tanh` | `tl.exp() / tl.tanh()` | 数学内建 |
| `linalg.reduce` | `tl.sum() / tl.max()` | 归约函数 |
| `linalg.matmul` | `tl.dot()` | Tensor Core |
| `linalg.broadcast` | 自动广播 | 标量→张量 |

## Triton 是怎么变成 MLIR 的

Triton 代码不是"直接对应"一个 MLIR 文件——它是经过完整编译器流水线**降低**成 MLIR 的。
以下是 SiLU kernel 的从 Python 到 LLVM 的全流程演示。

### 示例：SiLU 激活函数

```python
# Triton Python 源码 (30_silu.py)
@triton.jit
def silu_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = x * tl.sigmoid(x)    # ← 这是你要写的代码
    tl.store(Y + offsets, y)
```

### 降低路径

```
Triton Python
  │ @triton.jit 装饰器触发编译
  ▼
TTIR (Triton IR)                    ← 类似 MLIR 的高级 IR
  │ tl.sigmoid(x) 展开为 1/(1+exp(-x))
  │ x * sigmoid(x) → mul + div + exp
  ▼
TritonGPU                           ← 添加 GPU 线程/内存映射
  │ 决定 BLOCK 内线程布局
  │ 内存操作映射到 shared/global memory
  ▼
TritonToLinalg (转换到 MLIR)        ← 变成 MLIR linalg dialect
  │ tt.load  → memref.load + affine
  │ tt.store → memref.store
  │ 算术操作 → arith.mulf / arith.addf / math.exp
  ▼
MLIR (linalg + arith + math)        ← 对应 test-cases/mlir/
  │ 等价于: 02_intermediate/02_silu.mlir
  │ linalg.generic + arith.mulf + math.exp + arith.divf + arith.addf
  ▼
LLVM IR                             ← 通过 mlir-opt 降级
  │ --convert-linalg-to-affine-loops
  │ --lower-affine --convert-scf-to-cf
  │ --convert-func-to-llvm
  ▼
PTX (GPU 指令)                      ← Triton 再转 LLVM → PTX
  │ NVIDIA 编译器最终生成 SASS
  ▼
GPU 执行
```

### 关键点

1. **Triton 是一层更高级的 DSL**——你写的 Python 被 @triton.jit 编译，经过多层 IR 降低。
2. **本项目的 MLIR 用例**对应 Triton 降低路径中的**一段**（TritonToLinalg 之后的 linalg IR），而不是 Triton 源码本身。
3. **差异**：Triton 的 `tl.sigmoid(x)` 是内建函数，在 TTIR 阶段展开为 `1/(1+exp(-x))`；本项目的 MLIR 用例直接写这 4 步（negf→exp→addf→divf）——这恰好是 Triton 展开后的样子。
4. **两者的关系**：
   - Triton 源码：你**写什么**（高层语义）
   - MLIR linalg：编译器**展开成什么**（中间表示）
   - LLVM IR：编译器**降低成什么**（接近机器码）

### 其他算子对比

| Triton 一行 | 展开为 MLIR 步数 | 本项目的 MLIR 文件 |
|------------|-----------------|-------------------|
| `tl.tanh(x)` | 1 步 (`math.tanh`) | `01_basic/03_tanh.mlir` |
| `tl.sigmoid(x)` | 4 步 (`negf+exp+addf+divf`) | `02_intermediate/01_sigmoid.mlir` |
| `tl.maximum(x, 0)` | 2 步 (`cmpf+select`) | `01_basic/02_relu.mlir` |
| `tl.sum(x)` | 1 步 (`linalg.reduce`) | `02_intermediate/07_reduce_sum.mlir` |
| `tl.dot(a, b)` | 1 步 (`linalg.matmul` → 74行LLVM) | `03_advanced/01_matmul.mlir` |

详见 `MAPPING.md`。
64|