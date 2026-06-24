# Phase 3 & 4 配套项目设计（修订版：结合 AscendNPU-IR）

> 替换 `plans/phase3-4-projects-design.md`
> 原则：能用 AscendNPU-IR 现成的不自己造

---

## 现状扫描

| 资源 | 内容 | 可复用程度 |
|------|------|-----------|
| `projects/hello-pass/` | Phase 2 已完成的 LLVM Pass | 作为模板 |
| AscendNPU-IR `bishengir/test/` | 131 个 .mlir 测试用例 | 全部可用 |
| AscendNPU-IR `bishengir/test/Conversion/convert-linalg-to-hfusion.mlir` | 22.9KB 融合测试 | 核心素材 |
| AscendNPU-IR `bishengir/test/HivmToLLVM/` | Lowering 测试 | 全程追踪素材 |
| 用户自己的 `ascendnpu-ir` fork | 同 AscendNPU-IR 结构 | 相同 |

---

## 结论：只需新建 1 个独立项目，其余指向 AscendNPU-IR

| 原计划 | 修订后 | 原因 |
|--------|--------|------|
| `projects/mlir-hello/` | ✅ **保留，新建** | 没有现成等价物，对标 hello-pass |
| `projects/mlir-fusion-demo/` | ❌ **去掉** | AscendNPU-IR 已有 22.9KB 真实测试用例 |
| `projects/ascend-trace/` | ❌ **去掉** | AscendNPU-IR 已有 131 个测试覆盖全程 |

---

## 项目 1：`projects/mlir-hello/` — MLIR 版 HelloPass（新建）

**为什么必须新建**：Phase 2 的 hello-pass 给了学习者"一键运行"的体验，Phase 3 需要同样的入门动作。AscendNPU-IR 没有这样最小化的独立 Pass 示例。

### 和 hello-pass 的精确对应

| | hello-pass (LLVM) | mlir-hello (MLIR) |
|---|---|---|
| 遍历单位 | `Function &F` | `func::FuncOp` |
| 回调函数 | `run(Function &F)` | `runOnOperation()` |
| 打印函数名 | `F.getName()` | `func.getName()` |
| 统计子结构 | `BB.size()` (基本块数) | `func.walk()` (Operation 数) |
| 注册方式 | `llvmGetPassPluginInfo()` | `mlirGetPassPluginInfo()` |
| 运行命令 | `opt --passes="hello"` | `mlir-opt --pass-pipeline="hello-mlir"` |
| 构建产物 | `libHelloPass.so` | `libHelloMLIRPass.so` |

**5 个文件**（CMakeLists.txt + HelloMLIRPass.cpp + test.mlir + run.sh + README.md），约 120 行。

**README 中必须有对照表**，让学习者直观感受 LLVM→MLIR 的迁移。

### 构建验证

和 hello-pass 一样在 macOS LLVM 22（Homebrew keg-only）下验证 `mlir-opt --load-pass-plugin` 能跑通。

---

## 替代方案 2：指向 AscendNPU-IR 测试用例（替代 mlir-fusion-demo）

不需要新建项目。在 `docs/ascend/01-husion-hivm-Dialect详解.md` 中直接指向真实测试文件：

```markdown
## 动手：看真实的融合测试

AscendNPU-IR 项目里有一个 22.9KB 的融合测试文件，
覆盖了 add/mul/sub/relu 等数十种融合模式：

**文件**：[bishengir/test/Conversion/convert-linalg-to-hfusion.mlir](https://github.com/Ascend/AscendNPU-IR/blob/main/bishengir/test/Conversion/convert-linalg-to-hfusion.mlir)

**在本地查看**：
\`\`\`bash
cd AscendNPU-IR
cat bishengir/test/Conversion/convert-linalg-to-hfusion.mlir | head -100
\`\`\`

**如何读这个文件**：
- 每个 `func.func` 是一个独立的测试用例
- `CHECK:` 注释描述预期输出
- 找到 `// -----` 分隔线，上下对照输入和预期输出
```

**优势**：学习者看的是真实工业代码，不是教学 demo。

---

## 替代方案 3：指向 AscendNPU-IR 的 Lowering 全路径（替代 ascend-trace）

不需要新建项目。在 `docs/ascend/02-一个Ascend-Pass详解.md` 中展示真实调用链：

```markdown
## 看一条指令走完全程

在 AscendNPU-IR 中，用 `mlir-opt` 的 `--mlir-print-ir-after-all` 看每个 Pass 的输出：

\`\`\`bash
cd AscendNPU-IR/build
./bin/ascendnpu-ir-opt \
  --mlir-print-ir-after-all \
  --pass-pipeline="builtin.module(
    convert-linalg-to-hfusion,
    convert-hfusion-to-hivm,
    convert-hivm-to-llvm
  )" \
  ../bishengir/test/Conversion/convert-linalg-to-hfusion.mlir \
  2>&1 | head -200
\`\`\`

你会看到 IR 的完整演变：
linalg.generic → husion.elemwise_binary → hivm.vadd → llvm.func
```

---

## 最终交付

| 项目 | 类型 | 说明 |
|------|------|------|
| `projects/mlir-hello/` | 新建，可运行 | MLIR 版 HelloPass，对标 hello-pass |
| AscendNPU-IR 测试用例 | 文档中引用 | fusion-demo 和 ascend-trace 改为指向真实代码 |

**不需要新建** `mlir-fusion-demo` 和 `ascend-trace`——AscendNPU-IR 已经有更好的真实素材。

---

## 和当前项目的关系

```
ascend-npu-compiler-learning (本项目)
├── projects/
│   ├── hello-pass/        ← Phase 2 动手 ✅
│   └── mlir-hello/        ← Phase 3 动手 🚧 新建
│
└── docs/ascend/           ← Phase 4 文档
    ├── 01-... 指向 ──→ AscendNPU-IR bishengir/test/  (131 个测试)
    └── 02-... 指向 ──→ AscendNPU-IR 的 Lowering 全路径

AscendNPU-IR (外部项目，clone 后即可使用)
└── bishengir/test/        ← 131 个 .mlir 测试用例
    ├── Conversion/convert-linalg-to-hfusion.mlir  (22.9KB 融合)
    └── HivmToLLVM/                                (Lowering 全路径)
```

**学习流程**：先在本项目跑 hello-pass → 再跑 mlir-hello → 然后 clone AscendNPU-IR → 用其测试用例学真实代码。
