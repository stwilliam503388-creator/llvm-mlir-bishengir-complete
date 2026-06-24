# 动手项目

本目录包含与学习路线各阶段配套的动手实践项目。

## 项目列表

| 项目 | 难度 | 对应阶段 | 说明 | 状态 |
|------|------|---------|------|------|
| [hello-pass](./hello-pass/) | ⭐ | Phase 2 LLVM | 第一个 LLVM Pass：打印函数信息 | ✅ |
| [mlir-hello](./mlir-hello/) | ⭐ | Phase 3 MLIR | MLIR 版 HelloPass，45 行独立程序 | ✅ |
| [ascend-samples](./ascend-samples/) | ⭐⭐ | Phase 4 Ascend | 5 个 IR Lowering 精选用例 | ✅ |
| [opt-pass](./opt-pass/) | ⭐⭐ | Phase 2 进阶 | 死代码消除 Pass，35 行 | ✅ |

## 快速开始

```bash
# LLVM Pass — 一键运行
cd hello-pass && ./run.sh

# MLIR Pass — 一键运行
cd mlir-hello && ./run.sh

# Ascend 用例 — 需先构建 AscendNPU-IR
cd ascend-samples && cat 01-simple-add/README.md
```

## 编写指南

- 提供最小、最直接的功能复现
- 清晰直白的注释
- 通过文件树展示清晰的项目结构
- 必要的测试用例和预期输出
- 提供一键运行的 `run.sh` 脚本

## 与 LLVM 官方示例的区别

本项目将 LLVM 和 MLIR 的官方示例重新组织和改编为更易于初学者理解的版本，并根据需要补充背景知识。Ascend-samples 的用例来自 AscendNPU-IR 官方项目的 131 个测试，精选 5 个关键用例并附逐行解读。
