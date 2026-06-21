---
title: Triton接口文档分析
source: ascendnpu-ir/docs/zh_cn/developer_guide/conversion/triton_interface.md

tags:
  - 工程
---

# Triton接入 — 文档分析

**源文件**: `docs/source/zh_cn/developer_guide/conversion/triton_interface.md`
**在线**: https://ascendnpu-ir.gitcode.com/zh_cn/developer_guide/conversion/triton_interface.html
**总行数**: 1170

## 文档定位

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) 是一个协助 Triton 接入 Ascend 平台的重要组件。完成 Triton Ascend 的构建与安装后，使用者在执行 Triton 算子时，即可选用 Ascend 作为后端。

## 安装与执行

当前Triton-Ascend要求的Python版本为：**py3.9-py3.11**。

子章节:
- 环境准备
- 调用Triton Kernel

*(39 行)*

## Triton Op到Ascend NPU IR Op的转换

Triton Ascend将Triton方言的高级GPU抽象操作逐级下降为Linalg、HFusion和HIVM等目标方言，最终生成可在Ascend NPU上高效执行的优化中间表示。下表详细列出了各类Triton操作与其在下降过程中所对应的Ascend NPU IR操作。

*(49 行)*

## Triton拓展操作

AscendNPU-IR增量提供了语言特性，其中Triton-Ascend基于NPU IR扩展了一部分操作，若要使能相关能力，你需要import以下的模块。

子章节:
- 同步与调试操作
- 硬件查询与控制操作
- 编译优化提示
- 张量切片操作
- 张量计算操作
- 索引与收集操作

*(429 行)*

## Triton独有定制化操作

在A5的架构下，Triton-Ascend的Custom Op支持用户自行定制操作并使用它。定制操作在运行时转换为对设备侧实现函数的调用，可以调用已有的库函数，也可以调用由用户提供的源码或字节码编译生成的实现函数。

子章节:
- 基本用法
- 内置定制操作
- 参数有效性检查
- 输出参数和返回值
- 调用函数的符号名
- 源码与编译
- 参数转换规则
- 常量参数类型
- 封装定制操作

*(183 行)*

## Triton独有扩展枚举

|--------|----------|

子章节:
- SYNC_IN_VF
- PIPE

*(32 行)*

## Triton Op转换 — 核心API

关键概念:
- 程序信息类 Op
- 归约类 Op
- 张量操作类 Op
- 数值计算类 Op
- 指针运算类 Op
- 访存类 Op
