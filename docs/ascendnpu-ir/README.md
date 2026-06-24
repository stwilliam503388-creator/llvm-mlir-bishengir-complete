# AscendNPU-IR 相关文档

本目录整理 AscendNPU-IR / BishengIR 相关资料，包括官方文档翻译、dialect/pass 说明和本项目中的对照入口。

> 💡 术语不懂：查 [技术术语速查手册](../reference/技术术语速查手册.md)，其中包含 `hfusion`、`hivm`、`Cube`、`CANN`、`ConversionTarget` 等 AscendNPU-IR 常见术语。

## 当前目录

```text
docs/ascendnpu-ir/
├── README.md
└── translations/        # 官方文档翻译与分析笔记
```

## translations 内容范围

| 类型 | 示例 | 用途 |
|---|---|---|
| Dialect 文档 | `HFusionDialect.md`、`HIVMDialect-翻译-上.md`、`HACCDialect.md` | 理解 AscendNPU-IR 自定义 IR 层级 |
| Pass 文档 | `HFusionPasses-翻译.md`、`HIVMPasses-翻译.md`、`HACCPasses.md-翻译.md` | 理解 lowering 和转换流程 |
| 接口/开发指南 | `Triton接口文档分析.md`、`developer_guide-passes-dialects分析.md` | 对照 Triton 接入与 pass/dialect 开发 |
| 扩展 dialect | `MathExtDialect.md-翻译.md`、`MemRefExtDialect.md-翻译.md` | 理解标准 MLIR dialect 的扩展点 |

## 与本仓库其他内容的关系

| 想了解 | 推荐入口 |
|---|---|
| Ascend NPU 硬件和后端概念 | [docs/ascend/README.md](../ascend/README.md) |
| MLIR 基础和 Toy Tutorial | [docs/mlir/README.md](../mlir/README.md) |
| Ascend Lowering 精选用例 | [projects/ascend-samples](../../projects/ascend-samples/) |
| 标准 MLIR 模拟 AscendNPU-IR 降级 | [projects/ascendnpu-ir-demo](../../projects/ascendnpu-ir-demo/) |
| 自定义 Pass 参考代码 | [projects/ascendnpu-ir-op-counter](../../projects/ascendnpu-ir-op-counter/) |
| 本项目文件与外部源码的映射 | [references/ascendnpu-ir-mapping.md](../../references/ascendnpu-ir-mapping.md) |

## 建议阅读顺序

1. 先读 [docs/ascend/00-Ascend-NPU硬件概述.md](../ascend/00-Ascend-NPU硬件概述.md)。
2. 再读 [docs/ascend/01-husion-hivm-Dialect详解.md](../ascend/01-husion-hivm-Dialect详解.md)。
3. 跑或阅读 [projects/ascendnpu-ir-demo](../../projects/ascendnpu-ir-demo/) 中的 `01_vecadd.mlir`。
4. 需要深入某个 dialect/pass 时，再查 `translations/` 中的对应文档。
