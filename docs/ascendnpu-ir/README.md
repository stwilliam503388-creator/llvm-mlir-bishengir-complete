# AscendNPU-IR 相关文档

> 💡 **术语不懂？** → 查 `参考/技术术语速查手册.md`（`docs/reference/`），有"hfusion/hivm/Cube/CANN/ConversionTarget"等 bishengir 专用术语的解释。

## 目录说明

```
docs/ascendnpu-ir/
├── README.md               ← 本文件
├── translations/           ← 9 篇中文翻译文档（来自官方文档）
│   ├── 01-AnnotationPasses.md     — Annotation dialect 转换 Pass
│   ├── 02-HACCPasses.md           — HACC dialect 转换 Pass
│   ├── 03-ScopePasses.md          — Scope dialect 转换 Pass
│   ├── 04-SymbolPasses.md         — Symbol dialect 转换 Pass
│   ├── 05-AnnotationDialect.md    — Annotation dialect 定义
│   ├── 06-ScopeDialect.md         — Scope dialect 定义
│   ├── 07-SymbolDialect.md        — Symbol dialect 定义
│   ├── 08-MathExtDialect.md       — 数学扩展 dialect
│   └── 09-MemRefExtDialect.md     — 内存引用扩展 dialect
└── analysis/               ← 2 篇深度分析笔记
    ├── BishengIR代码仓库解读.md    — 代码仓库逐目录解读
    └── AscendNPUIR文档总结.md      — 官方文档体系总结
```

## 与源码的对应

| 文档 | 对应 ascendnpu-ir 源码路径 |
|------|--------------------------|
| 翻译文档 | `bishengir/docs/cn/` |
| 代码仓库解读 | `bishengir/` 全目录 |
| 文档总结 | `bishengir/docs/` 全目录 |

## 与本项目笔记的对应

| 这些文档 | 对应本项目笔记 |
|---------|--------------|
| 翻译文档 | `docs/mlir/L00-速通与bishengir实战.md` |
| 代码仓库解读 | `docs/mlir/L06-TritonMLIR体系分析.md`, `L07-triton-ascend后端分析.md` |
| 文档总结 | `docs/mlir/L00.md` + `references/ascendnpu-ir-mapping.md` |

## 先读哪个

1. 先读 `analysis/AscendNPUIR文档总结.md` — 了解文档全貌
2. 再读 `analysis/BishengIR代码仓库解读.md` — 了解代码仓库结构
3. 需要深入了解某个 dialect → 读对应的翻译文档
