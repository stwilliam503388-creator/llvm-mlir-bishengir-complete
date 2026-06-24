# Phase 4 — Ascend NPU 编译器后端

> 衔接 Phase 3（MLIR），进入 Ascend NPU 编译器实战
> 前置：[Phase 3 MLIR](../mlir/README.md)

## 学习路径

```
① NPU 硬件概述 ──→ ② Dialect 详解 ──→ ③ Pass 实战 ──→ ④ 构建调试
    (10 min)         (20 min)           (25 min)        (30 min)
```

## 文档列表

| # | 文档 | 目标 | 时间 |
|---|------|------|------|
| 00 | [Ascend NPU 硬件概述](./00-Ascend-NPU硬件概述.md) | Da Vinci 架构 + NPU vs GPU | 10 min |
| 01 | [husion / hivm Dialect 详解](./01-husion-hivm-Dialect详解.md) | 融合→指令 完整 Lowering | 20 min |
| 02 | [一个 Ascend Pass 详解](./02-一个Ascend-Pass详解.md) | .td + .cpp + 测试 三层拆解 | 25 min |
| 03 | [构建与调试指南](./03-构建与调试指南.md) | clone → 构建 → 跑测试 → 调试 | 30 min |

## 动手项目

| 项目 | 说明 |
|------|------|
| **[ascend-samples](../../projects/ascend-samples/)** | 从 AscendNPU-IR 131 个测试中精选 5 个关键用例 |

## 与 Phase 3 的衔接

| Phase 3 学会的 | Phase 4 怎么用到 |
|---------------|----------------|
| MLIR Dialect 概念 | husion 和 hivm 的具体设计 |
| Toy Tutorial 的 Pass | ConvertLinalgToHusion 的真实实现 |
| ascendnpu-ir 项目结构 | 深入源码 + 构建运行 |

## 依托资源

- [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) — 华为官方开源项目
- [华为 Ascend 社区](https://www.hiascend.com/) — 硬件文档和 CANN 软件栈

## 学完验证

- [ ] 能描述 Da Vinci 架构的 Cube/Vector/Scalar 三单元
- [ ] 能画出 linalg → husion → hivm 完整路径
- [ ] 能看懂一个 Pass 的 .td + .cpp + 测试 三部分
- [ ] 能在自己机器上构建并运行 ascendnpu-ir 测试
