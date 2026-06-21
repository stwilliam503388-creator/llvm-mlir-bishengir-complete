# projects/ — 工程实践项目

> **声明**：本目录下所有工程均为**独立的学习/教学项目**，非 [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) 官方出品，不包含其源码。

---

## 与 AscendNPU-IR 的关系

| 工程 | 与 AscendNPU-IR 的关系 | 性质 |
|------|----------------------|------|
| **ascendnpu-ir-demo** | 用标准 `mlir-opt` 模拟 AscendNPU-IR 三阶段降级流水线 (Linalg → HFusion → HIVM) | 教学演示 |
| **ascendnpu-ir-op-counter** | 为 AscendNPU-IR dialect 编写的参考 Pass 代码，需要 AscendNPU-IR 构建系统才能编译 | 参考实现 |
| **standalone-mlir** | 独立的 MLIR dialect 项目，对照表中提到 AscendNPU-IR，但代码完全独立 | 独立教学项目 |
| **toy-mini** | 纯 C++17 Toy 语言解析器，对应 LLVM Toy Tutorial，与 AscendNPU-IR 无直接关系 | 独立教学项目 |

---

## 命名说明

AscendNPU-IR 是华为官方仓库名称（https://github.com/Ascend/AscendNPU-IR），其源码内部使用 "BishengIR"（毕昇 IR）作为编译器核心组件代号（命名空间 `bishengir-opt`、目录 `bishengir/`）。

本项目中：
- 工程目录使用 `ascendnpu-ir-` 前缀 — 对齐官方品牌名
- 源码文件仍保留 `Bishengir` 前缀（如 `BishengirOpCounter.cpp`）— 对齐 AscendNPU-IR 源码中的实际命名

---

## 快速开始

```bash
# ascendnpu-ir-demo: 模拟降级流水线
cd ascendnpu-ir-demo && bash run-demo.sh

# toy-mini: 零依赖 Toy 解析器
cd toy-mini && g++ -std=c++17 -o toymini toymini.cpp && ./toymini

# standalone-mlir: 需要 LLVM/MLIR 安装
cd standalone-mlir && cmake -S . -B build && cmake --build build
```
