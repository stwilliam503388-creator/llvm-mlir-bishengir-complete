# ascendnpu-ir-demo 自动化测试说明

本文档说明 `ascendnpu-ir-demo` 的测试组织、运行方式和验证原则。

## 快速开始

```bash
cd projects/ascendnpu-ir-demo

# 运行全部 MLIR 用例
bash run-tests.sh

# 详细模式
bash run-tests.sh --verbose

# 只运行名称匹配的用例
bash run-tests.sh vecadd
bash run-tests.sh advanced
```

## 用例组织

```text
test-cases/mlir/
├── 01_basic/          # ⭐ 入门：10 个
├── 02_intermediate/   # ⭐⭐ 进阶：11 个
└── 03_advanced/       # ⭐⭐⭐ 复杂：10 个
```

| 难度 | 数量 | 涉及概念 | 示例算子 |
|---|---:|---|---|
| ⭐ 入门 | 10 | `linalg.generic`、parallel iterator、逐元素操作 | vecadd、relu、tanh、broadcast、fill、fused |
| ⭐⭐ 进阶 | 11 | reduction iterator、组合模式、条件分支 | sigmoid、silu、gelu、reduce_sum、reduce_max、layer_norm |
| ⭐⭐⭐ 复杂 | 10 | `linalg.matmul`、conv/pooling、多步 pipeline | matmul、gemm_relu、conv2d、depthwise_conv、batch_norm |

总计 **31 个 MLIR 用例**。另有 `test-cases/triton/` 下 **28 个 Triton 对照 kernel**。

## 测试文件格式

每个 `.mlir` 文件自包含测试指令，采用 LLVM lit / FileCheck 风格：

```mlir
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for

module {
  func.func @vecadd(...) { ... }
}
```

| 标注 | 含义 |
|---|---|
| `// RUN:` | 要执行的命令 |
| `%s` | 当前测试文件路径 |
| `%S` | 当前测试文件所在目录 |
| `// CHECK:` | FileCheck 风格的期望输出 |
| `// CHECK-RUN1:` | `run-tests.sh` 针对第 1 条 RUN 命令的轻量检查 |

## 运行模式

### 有 `mlir-opt`

`run-tests.sh` 会执行每个 `.mlir` 文件中的 `// RUN:` 命令。

```bash
export MLIR_OPT=/opt/homebrew/opt/llvm/bin/mlir-opt
bash run-tests.sh
```

如果系统没有 `FileCheck`，脚本会对含 `| FileCheck` 的命令只执行管道左侧，以验证 lowering 命令本身可以运行。

### 无 `mlir-opt`

脚本自动切换到标注验证模式，只检查每个 `.mlir` 文件是否存在 `// RUN:` 标注。这让仓库在轻量 CI 或未安装 LLVM/MLIR 的机器上仍可做基础结构验证。

## 与 AscendNPU-IR 的对照

```text
AscendNPU-IR 流水线:                    标准 MLIR 等效演示:
─────────────────────────              ────────────────────────
-convert-linalg-to-hfusion       ↔      --convert-linalg-to-affine-loops
-convert-arith-to-hfusion        ↔      --lower-affine
-convert-hfusion-to-hivm         ↔      --convert-scf-to-cf / --convert-func-to-llvm
```

## 如何添加新用例

1. 在对应难度目录下添加 `.mlir` 文件。
2. 文件头部说明功能、公式和对应的 AscendNPU-IR 语义。
3. 添加至少一条 `// RUN:` 标注。
4. 可选添加 `// CHECK:` 或 `// CHECK-RUN<N>:` 标注。
5. 运行 `bash run-tests.sh <关键词>` 验证。

## 相关脚本

| 脚本 | 功能 |
|---|---|
| `run-tests.sh` | 自动发现并验证 `test-cases/mlir/*/*.mlir` |
| `run-demo.sh` | 对每个用例输出三阶段 lowering 结果到 `results/` |
| `variants/compare.sh` | 对比 matmul 的 4 种优化方案 |
