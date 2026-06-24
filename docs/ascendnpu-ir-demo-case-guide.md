# ascendnpu-ir-demo 用例导读

本文迁移并整理原 `projects/ascendnpu-ir-demo/README.md` 中较长的教学说明。项目 README 只保留运行入口；每个算子为什么重要、对应什么 AI 场景、映射到什么 MLIR / AscendNPU-IR 概念，集中在这里说明。

## 用例分级

`projects/ascendnpu-ir-demo/test-cases/mlir/` 下共有 31 个 MLIR 用例，按学习难度分为三组：

| 难度 | 数量 | 目录 | 核心概念 | 建议读者 |
|---|---:|---|---|---|
| ⭐ 入门 | 10 | `01_basic/` | `linalg.generic`、parallel iterator、逐元素运算、broadcast、fill | 读完 Primer 即可 |
| ⭐⭐ 进阶 | 11 | `02_intermediate/` | reduction iterator、组合模式、条件分支、归一化片段 | 已理解 LLVM/MLIR 基本 IR |
| ⭐⭐⭐ 复杂 | 10 | `03_advanced/` | matmul、conv、pooling、多步 pipeline、named op 语义 | 正在学习 lowering / dialect |

配套的 Triton 对照位于 `projects/ascendnpu-ir-demo/test-cases/triton/`。当前有 28 个 Triton kernel，3 个 legacy MLIR 用例复用已有 vecadd、fused、matmul 对照。

## 第一梯队：入门 ⭐

入门用例主要训练 `linalg.generic` 直觉：输入张量、输出张量、indexing map、iterator 类型、region 中的标量计算。

| 用例 | AI 场景 | MLIR 重点 | AscendNPU-IR 类比 |
|---|---|---|---|
| `01_vecadd.mlir` | residual add / shortcut | `linalg.generic` + `arith.addf` | `hfusion.elemwise_binary {fun = add}` |
| `02_relu.mlir` | CNN / FFN 激活 | `arith.cmpf` + `arith.select` | `hfusion.elemwise_unary {fun = relu}` |
| `03_tanh.mlir` | RNN / LSTM 激活 | `math.tanh` | unary math op 映射 |
| `04_softmax_exp.mlir` | Attention softmax 的 exp 片段 | `math.exp` | 逐元素 math 操作 |
| `05_broadcast.mlir` | bias、位置编码、BatchNorm 参数广播 | 标量到矩阵的 affine map | `hfusion.broadcast` |
| `06_dropout.mlir` | 训练期 dropout 的缩放片段 | `arith.mulf` | elemwise mul |
| `07_fill.mlir` | 输出缓冲区初始化 | `yield %cst` | `linalg.fill` / fill 语义 |
| `08_fused.mlir` | add + mul 算子融合 | 连续两个 `linalg.generic` | HFusion 的融合动机 |
| `09_vecadd_128.mlir` | 更大向量残差加法 | 与 vecadd 相同，观察规模变化 | 复用 add 语义 |
| `10_fused_128.mlir` | 更大向量融合操作 | 连续 elementwise pipeline | 融合收益更明显 |

学习重点：先不要急着理解所有 lowering pass，先确认自己能看懂 `ins` / `outs` / `indexing_maps` / `iterator_types` / `linalg.yield`。

## 第二梯队：进阶 ⭐⭐

进阶用例开始出现组合函数、条件选择、reduction 和归一化片段。它们更接近真实模型里的算子图，而不是单一标量操作。

| 用例 | AI 场景 | MLIR 重点 | 学习提示 |
|---|---|---|---|
| `01_sigmoid.mlir` | 二分类输出、RNN 门控 | `negf` + `exp` + `divf` | 观察多步 elementwise 如何组合 |
| `02_silu.mlir` | LLaMA / Mistral / Gemma FFN | sigmoid + multiply | 现代 LLM 常见门控激活 |
| `03_leaky_relu.mlir` | GAN / 传统 CNN | compare + select | 条件分支的最小例子 |
| `04_gelu_tanh.mlir` | BERT / GPT 系列激活 | tanh 近似 GELU | 组合 math op |
| `05_hard_sigmoid.mlir` | MobileNet 轻量化激活 | maximum / minimum | 分段线性近似 |
| `06_prelu.mlir` | 图像超分辨率 | 可训练参数参与 elementwise | 参数张量参与计算 |
| `07_reduce_sum.mlir` | LayerNorm 均值 | `reduction` iterator | 第一个 reduction 重点看这里 |
| `08_reduce_max.mlir` | softmax 数值稳定 | max reduction + select | 理解归约初值和比较 |
| `09_softmax_complete.mlir` | Attention softmax 片段 | reduction + alloc + store | 多步 pipeline |
| `10_clamp.mlir` | 梯度裁剪、量化友好截断 | 两次 compare/select | 分段函数 |
| `11_layer_norm.mlir` | Transformer LayerNorm | sub / mul / scale / bias | 与 reduction 配合才是完整 LN |

学习重点：看懂 `parallel` 与 `reduction` iterator 的区别。前者每个输出元素独立计算；后者多个输入元素汇聚到同一个输出。

## 第三梯队：复杂 ⭐⭐⭐

复杂用例展示了为什么 NPU 编译器不能过早把高层语义展开成标量循环。matmul、conv、pooling 这类操作如果完全 lowering 到标量级 IR，会迅速膨胀；真实后端更希望保留高层语义并映射到硬件单元。

| 用例 | AI 场景 | MLIR 重点 | AscendNPU-IR 视角 |
|---|---|---|---|
| `01_matmul.mlir` | Linear / MLP / QKV projection | `linalg.matmul` named op | `hfusion.cube_matmul` → `hivm.mmul` |
| `02_gemm_relu.mlir` | GEMM + activation 融合 | matmul 后接 elementwise | 融合减少中间 buffer |
| `03_depthwise_conv.mlir` | MobileNet / EfficientNet | depthwise conv named op | 保留卷积语义便于硬件映射 |
| `04_conv2d.mlir` | CNN / YOLO / UNet | reduction + affine map | 模式匹配到卷积语义 |
| `05_max_pool.mlir` | CNN 下采样 | 手写 affine loop + max | pooling 语义 |
| `06_avg_pool.mlir` | ResNet 分类前下采样 | affine loop + sum/div | pooling + reduction |
| `07_global_avg_pool.mlir` | 分类头 | 全图 reduction | 参数量为 0 的聚合 |
| `08_batch_norm_part1.mlir` | BatchNorm 均值 | reduction + parallel 混合 | reduce + broadcast 组合 |
| `09_batch_norm_part2.mlir` | BatchNorm 归一化 | 多输入 `linalg.generic` | elemwise 组合 |
| `10_matmul_4x4x4.mlir` | legacy matmul 对照 | 小尺寸 matmul | 与 matmul Triton 对照复用 |

学习重点：复杂算子不是为了“手写循环”，而是为了观察“如果不保留 matmul/conv 高级语义，IR 会如何膨胀”。

## matmul 膨胀与优化路线

标准 MLIR 把 matmul 降级成 affine/scf/llvm 后，会显式出现循环、load、mul、add、store。这个过程适合教学，但不是 NPU 后端最终想要的形态。

```text
标准 MLIR 路径:
  linalg.matmul
      ↓
  affine.for × 3
      ↓
  scf/cf + arith
      ↓
  llvm.load / llvm.fmul / llvm.fadd / llvm.store

AscendNPU-IR 思路:
  linalg.matmul
      ↓
  hfusion.cube_matmul
      ↓
  hivm.mmul
      ↓
  Ascend Cube 单元执行
```

`projects/ascendnpu-ir-demo/variants/compare.sh` 用 4 个变体演示这个差异：

| Variant | 策略 | 核心思想 |
|---|---|---|
| V0 | baseline | 直接 lowering，观察标量循环展开 |
| V1 | tiling | 增加 tile 层级，改善局部性 |
| V2 | vectorize | 引入向量语义，减少标量指令 |
| V3 | hardware mapping | 保留 `matmul` 高级语义，类比映射到 `hivm.mmul` |

关键结论：**AI 编译器后端的价值不只是把 IR 降到底，而是在合适阶段保留可映射到硬件的高级语义。**

## 推荐阅读顺序

```text
1. 先看 01_basic/01_vecadd.mlir
2. 再看 01_basic/08_fused.mlir
3. 接着看 02_intermediate/07_reduce_sum.mlir
4. 然后看 03_advanced/01_matmul.mlir
5. 最后运行 variants/compare.sh 对照 matmul 优化
```

如果没有 `mlir-opt`，仍然可以运行：

```bash
cd projects/ascendnpu-ir-demo
bash run-tests.sh
```

脚本会切换到标注检查模式，确认每个用例都保留了 `// RUN:` 教学入口。安装 LLVM/MLIR 后，再用 `MLIR_OPT=/path/to/mlir-opt bash run-tests.sh` 执行真实 lowering。

## 和其他文档的关系

| 想继续学 | 推荐入口 |
|---|---|
| 编译器基础概念 | `docs/primer/README.md` |
| MLIR dialect / lowering | `docs/mlir/README.md` |
| Ascend 后端语义 | `docs/ascend/README.md` |
| AscendNPU-IR 官方文档翻译 | `docs/ascendnpu-ir/README.md` |
| Triton 对照 kernel | `projects/ascendnpu-ir-demo/test-cases/triton/MAPPING.md` |
