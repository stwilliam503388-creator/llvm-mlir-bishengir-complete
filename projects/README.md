# projects/ — 工程实践项目

> **声明**：本目录下所有工程均为**独立的学习/教学项目**，非 [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) 官方出品，不包含其源码。

---

## 与 AscendNPU-IR 的关系

本项目中的各个子工程，均围绕华为昇腾 NPU 编译器 AscendNPU-IR 进行学习与实践。它们通过**模拟、对照、参考**等方式帮助开发者理解 AscendNPU-IR 的设计思想，但**不直接包含** AscendNPU-IR 的任何源码。

| 工程 | 中文说明 | 与 AscendNPU-IR 的关系 | 性质 |
|------|---------|----------------------|------|
| `ascendnpu-ir-demo` | 可运行降级流水线 | 用 `mlir-opt` **模拟** AscendNPU-IR 三阶段降级 | 教学模拟 |
| `ascendnpu-ir-op-counter` | 自定义 Pass | **参考** AscendNPU-IR 的 Pass 注册方式编写 | 独立代码 |
| `toy-mini` | Toy 解析器 | 从零实现，与 AscendNPU-IR 无关 | 纯学习 |
| `standalone-mlir` | 自建 Dialect | 独立 CMake 项目，学习 MLIR 基础设施 | 纯学习 |

---

## 命名说明

```
目录名:   ascendnpu-ir-*     ← 对齐官方品牌
源码文件: Bishengir*.cpp     ← 对齐 AscendNPU-IR 源码中的实际命名空间
Pass名:   --bishengir-*      ← AscendNPU-IR bishengir-opt 的命令行参数，不改
```

---

## 快速开始

```bash
# 1. ascendnpu-ir-demo: 运行降级流水线
cd ascendnpu-ir-demo && bash run-demo.sh

# 2. toy-mini: 编译 Toy 解析器
cd toy-mini && g++ -std=c++17 -o toymini toymini.cpp && ./toymini

# 3. standalone-mlir: 查看 TableGen 定义
cd standalone-mlir && cat include/standalone/StandaloneOps.td
```
