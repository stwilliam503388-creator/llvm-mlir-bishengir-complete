# bishengir-demo 自动化测试说明

本文档详细介绍 bishengir-demo 项目的自动化测试体系，包括测试框架、用例组织、运行方式及验证原理。

---

## 概览

本项目采用**类 LLVM lit/FileCheck 的测试模式**，测试指令直接嵌入在 `.mlir` 源文件中。核心测试脚本为 `run-tests.sh`，可自动发现并验证所有测试用例的 MLIR 降级流水线是否正常工作。

---

## 快速开始

```bash
# 运行全部测试
bash run-tests.sh

# 详细模式（显示每条 RUN 命令和输出）
bash run-tests.sh --verbose

# 只运行匹配关键词的用例
bash run-tests.sh vecadd
bash run-tests.sh advanced
```

---

## 测试用例组织

测试用例位于 `test-cases/` 目录，按难度分为三级：

```
test-cases/
├── 01_basic/          ← ⭐ 入门（8 个）
├── 02_intermediate/   ← ⭐⭐ 进阶（11 个）
└── 03_advanced/       ← ⭐⭐⭐ 复杂（9 个）
```

| 难度 | 数量 | 涉及概念 | 示例算子 |
|------|------|---------|---------|
| ⭐ 入门 | 8 | `linalg.generic` + parallel iterator | vecadd, relu, tanh, softmax, broadcast, dropout, fill, fused |
| ⭐⭐ 进阶 | 11 | reduction iterator、组合模式、条件分支 | sigmoid, silu, leaky_relu, gelu, prelu, reduce_sum/max, layer_norm |
| ⭐⭐⭐ 复杂 | 9 | `linalg.matmul`、named op、多步 pipeline | matmul, gemm_relu, conv2d, depthwise_conv, max/avg_pool, batch_norm |

总计 **28 个测试用例**。

---

## 测试文件格式

每个 `.mlir` 文件自包含测试指令，遵循 LLVM FileCheck 惯例：

```mlir
// RUN: mlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm %s
// CHECK: affine.for
// CHECK: arith.addf

module {
  func.func @vecadd(...) { ... }
}
```

### 标注说明

| 标注 | 含义 |
|------|------|
| `// RUN: <命令>` | 定义要执行的测试命令 |
| `%s` | 自动替换为当前文件路径 |
| `%S` | 自动替换为当前文件所在目录 |
| `// CHECK: <模式>` | 期望输出中必须包含的文本模式 |
| `// CHECK-RUN<N>: <模式>` | 针对第 N 条 RUN 命令的输出验证 |

---

## 验证流程（五阶段）

`run-tests.sh` 对每个用例依次执行以下验证：

| 阶段 | 验证内容 | 对应 MLIR Pass |
|------|---------|---------------|
| 1. 语法验证 | `mlir-opt` 能正确解析 `.mlir` 文件 | — |
| 2. Stage1 降级 | Linalg → Affine | `--convert-linalg-to-affine-loops` |
| 3. Stage2 降级 | Affine → SCF → CF | `--lower-affine --convert-scf-to-cf` |
| 4. Stage3 降级 | → LLVM Dialect（仅支持的用例） | `--convert-func-to-llvm` |
| 5. 输出验证 | CHECK 模式出现在输出中 | FileCheck 风格匹配 |

### 与 bishengir 的对照关系

```
bishengir 流水线:                   标准 MLIR 等效流水线:
─────────────────────────           ─────────────────────────
-convert-linalg-to-hfusion    ↔    --convert-linalg-to-affine-loops
-convert-arith-to-hfusion     ↔    --lower-affine
-convert-hfusion-to-hivm      ↔    --convert-scf-to-cf
```

---

## 环境要求与容错设计

### 有 `mlir-opt` 的环境

完整执行所有 RUN 命令并验证 CHECK 模式。

```bash
# 指定 mlir-opt 路径
export MLIR_OPT=/opt/homebrew/opt/llvm/bin/mlir-opt
bash run-tests.sh
```

### 无 `mlir-opt` 的环境

脚本自动切换为**标注验证模式**：仅检查 `.mlir` 文件中是否存在 `// RUN:` 标注，不执行实际编译命令。这确保了在 CI 或无 LLVM 环境中仍能进行基本验证。

---

## 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 全部测试通过 |
| `1` | 有测试失败 |

---

## 测试输出示例

```
╔══════════════════════════════════════════════════════════════╗
║  bishengir-demo: 自动化测试                                ║
╚══════════════════════════════════════════════════════════════╝

═══ 01_basic ═══
── 01_basic/01_vecadd ──
  ✓ PASS: 01_basic/01_vecadd
── 01_basic/02_relu ──
  ✓ PASS: 01_basic/02_relu
...

══════════════════════════════════════════════════════════════
  结果: 28 通过, 0 失败, 0 跳过 (共 28 个)
══════════════════════════════════════════════════════════════
```

---

## 如何添加新测试用例

1. 在对应难度目录下创建 `.mlir` 文件（如 `test-cases/01_basic/09_new_op.mlir`）
2. 文件头部添加注释说明（功能、公式、对应 bishengir 算子）
3. 添加 `// RUN:` 标注指定降级命令
4. 添加 `// CHECK:` 标注指定期望输出模式
5. 运行 `bash run-tests.sh new_op` 验证

### 模板

```mlir
// NewOp — 简要描述
// 公式: ...
// bishengir: hfusion.xxx

// RUN: mlir-opt --convert-linalg-to-affine-loops %s
// RUN: mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf %s
// CHECK: affine.for

module {
  func.func @new_op(...) {
    // MLIR IR ...
    return
  }
}
```

---

## 相关脚本

| 脚本 | 功能 |
|------|------|
| `run-tests.sh` | 自动化测试（验证正确性） |
| `run-demo.sh` | 演示三阶段降级（可视化中间结果，保存到 `results/`） |

---

## 设计理念

1. **自包含** — 测试指令嵌入 `.mlir` 文件，单文件即可理解和复现
2. **渐进分级** — 从简单逐元素算子到复杂卷积/归一化，方便学习
3. **环境自适应** — 有/无 `mlir-opt` 都能运行，不硬性依赖 LLVM 安装
4. **映射清晰** — 每个用例标注了对应的 bishengir 自定义 Pass 语义，便于理解定制编译器与标准 MLIR 的关系
