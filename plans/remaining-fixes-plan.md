# 收尾计划：补齐剩余 1 项

> Status: planning | Created: 2026-06-21

---

## 待补项

| # | 原始问题 | 当前状态 |
|---|---------|---------|
| 7 | 缺少"为什么学"动机章节 | ❌ |
| 9 | diagrams/ 引用清理 | ✅ 已无残留引用 |

---

## 唯一待做：动机章节

### 目标

零基础学习者打开项目后，第一个问题不是"编译器是什么"，而是"我为什么要花时间学这个"。在 Primer 之前需要一个 5 分钟读完的动机页。

### 位置

`docs/why-ascend.md` — 放在 primer 之前，README 和 quickstart 引用它。

### 内容大纲

```
1. 学编译器后端能干什么？（5 个真实场景）
   - 看懂 PyTorch 报错里的 IR dump
   - 给 AI 框架写自定义算子
   - 看懂 Triton kernel 编译流程
   - 为自家硬件写编译器后端
   - 参与开源编译器项目（LLVM/MLIR）

2. 为什么选 Ascend NPU？
   - 华为昇腾是国内最大的 AI 芯片生态
   - 岗位需求：CANN 开发、TBE 算子、编译器优化
   - 与 CUDA 生态的对比（学习曲线、社区、就业）
   - Ascend 编译器用的是 MLIR，学到的技能可迁移

3. 这个项目能带你走到哪？
   - Phase 1: 理解编译器基本概念
   - Phase 2: 能写 LLVM Pass
   - Phase 3/4 (计划中): MLIR + Ascend 后端实战
   - 对标岗位：编译器开发工程师、AI 框架开发、NPU 算子开发

4. 学习建议
   - 先跑通 hello-pass 建立信心
   - 概念不需要一次全懂，用到再查术语表
   - 预计总时间：Phase 1+2 约 2 小时
```

### 关联修改

| 文件 | 修改 |
|------|------|
| `docs/why-ascend.md` | 新建 |
| `docs/quickstart.md` | 开头加"先读 why-ascend"引导 |
| `docs/primer/README.md` | 开头加"已经知道为什么学？跳到这里" |
| `README.md` | 快速开始前加一句"不了解为什么学？→ why-ascend" |

### 设计约束

- ≤300 行
- 不堆砌行业数据，用场景化描述
- 末尾明确引导 → quickstart 或 primer
