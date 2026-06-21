# MLIR 测试用例

> 31 个 MLIR 测试用例，涵盖神经网络常见算子。
> 用 `mlir-opt` 降级到 LLVM IR，观察编译器如何做 Lowering。

## 目录结构

```
mlir/
│   ├── 01_basic/              8+2 个 (⭐ 入门)
│   │   ├── 01_vecadd.mlir     向量加法 / 残差连接
│   │   ├── 02_relu.mlir        ReLU 激活
│   │   ├── 03_tanh.mlir        Tanh 激活
│   │   ├── 04_softmax_exp.mlir Softmax 指数
│   │   ├── 05_broadcast.mlir   标量广播
│   │   ├── 06_dropout.mlir     Dropout
│   │   ├── 07_fill.mlir        常量填充
│   │   ├── 08_fused.mlir       算子融合演示 [新版本]
│   │   ├── 09_vecadd_128.mlir  向量加法 [原始版本, 128元素]
│   │   └── 10_fused_128.mlir   算子融合 [原始版本, 128元素]
│
├── 02_intermediate/       11 个 (⭐⭐ 进阶)
│   ├── 01_sigmoid.mlir     Sigmoid
│   ├── 02_silu.mlir        SiLU / Swish (LLaMA)
│   ├── 03_leaky_relu.mlir  Leaky ReLU
│   ├── 04_gelu_tanh.mlir   GELU (BERT)
│   ├── 05_hard_sigmoid.mlir Hard Sigmoid (MobileNet)
│   ├── 06_prelu.mlir       PReLU
│   ├── 07_reduce_sum.mlir  归约求和
│   ├── 08_reduce_max.mlir  归约最大值
│   ├── 09_softmax_complete.mlir Softmax 数值稳定
│   ├── 10_clamp.mlir       数值裁剪
│   └── 11_layer_norm.mlir  Layer Norm
│
│   └── 03_advanced/           9+1 个 (⭐⭐⭐ 复杂)
│       ├── 01_matmul.mlir      矩阵乘法 [新版本]
│       ├── 02_gemm_relu.mlir   GEMM + ReLU 融合
│       ├── 03_depthwise_conv.mlir 深度卷积
│       ├── 04_conv2d.mlir      二维卷积
│       ├── 05_max_pool.mlir    最大池化
│       ├── 06_avg_pool.mlir    平均池化
│       ├── 07_global_avg_pool.mlir 全局平均池化
│       ├── 08_batch_norm_part1.mlir BN 均值
│       ├── 09_batch_norm_part2.mlir BN 标准化
│       └── 10_matmul_4x4x4.mlir   矩阵乘法 [原始版本, 4x4]
```

## 难度说明

| 等级 | 前置知识 | 涉及 MLIR 概念 | 数量 |
|------|---------|---------------|------|
| ⭐ 入门 | 读完 Primer | `linalg.generic` + parallel | 8 |
| ⭐⭐ 进阶 | LLVM IR 基础 | `reduction` + 组合模式 | 11 |
| ⭐⭐⭐ 复杂 | MLIR dialect 概念 | named op + `affine.for` + 多步 pipeline | 9 |

## 如何运行

```bash
# 逐元素降级 (vecadd/relu/tanh 等)
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
  --convert-scf-to-cf --convert-func-to-llvm \
  mlir/01_basic/01_vecadd.mlir

# 矩阵乘法降级
mlir-opt --convert-linalg-to-affine-loops --lower-affine \
  --convert-scf-to-cf --convert-func-to-llvm \
  mlir/03_advanced/01_matmul.mlir

# 多路对比
bash projects/ascendnpu-ir-demo/variants/compare.sh
```

## 与 Triton 对应

每个 MLIR 用例有对应的 Triton 代码，详见 `../triton/`：

| MLIR 路径 | Triton 路径 |
|-----------|------------|
| `mlir/01_basic/01_vecadd.mlir` | `triton/01_basic/01_vecadd.py` |
| `mlir/02_intermediate/02_silu.mlir` | `triton/02_intermediate/10_silu.py` |
| `mlir/03_advanced/01_matmul.mlir` | `triton/03_advanced/20_matmul.py` |

完整映射见 `MAPPING.md` 和 `../triton/MAPPING.md`。

## 用例文件说明

每个 `.mlir` 文件头部包含：

```mlir
// 操作名 — 中文名
// 公式: (数学定义)
// 一句话: (大白话)
// 专业角色: (AI 中的作用)
// 用在哪: (具体模型/层)
// 降级: (MLIR 模式)
// bishengir: (对应关系)
module {
  // ... MLIR 代码
}
```

## 环境说明

使用 Homebrew LLVM 22.1.6 的 `mlir-opt`。部分 named op（`linalg.fill`、`linalg.pooling` 等）在 Homebrew 版本中未编译，已用 `linalg.generic` 或 `affine.for` 替代。详见 `../LIMITATIONS.md`。
