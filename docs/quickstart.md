# 快速入门

> 从零开始，用 2 小时建立 AI 编译器学习路线感。

如果你还不确定为什么要学，先花 5 分钟看：[为什么学 Ascend NPU 编译器？](./why-ascend.md)。

---

## 路线图

```text
① 环境检查 ──→ ② Primer 入门 ──→ ③ 第一个 LLVM Pass ──→ ④ 看一个 MLIR lowering demo
   (5 min)        (35 min)           (40-60 min)              (20 min)
```

---

## ① 环境检查

```bash
# 在项目根目录运行
bash setup.sh
```

看到 `✅ 所有核心依赖就绪` 即可直接运行 LLVM/MLIR 项目。若缺少 LLVM/MLIR，仍可先阅读文档，并运行不依赖工具链的检查：

```bash
bash scripts/check-docs.sh
```

---

## ② Primer 入门

按顺序阅读 [docs/primer/](./primer/)：

| # | 文档 | 重点 |
|---|---|---|
| 00 | [编译器是什么](./primer/00-编译器是什么.md) | 前端、优化、中后端 |
| 01 | [AST 与 IR](./primer/01-AST与IR.md) | 抽象语法树、中间表示、SSA |
| 02 | [Pass 与 Lowering](./primer/02-Pass与Lowering.md) | 编译器如何优化和降级 IR |
| 03 | [动手看 MLIR 长什么样](./primer/03-动手看MLIR长什么样.md) | 用 `mlir-opt` 观察 IR 变化 |
| 04 | [从 Triton 到 Ascend](./primer/04-从Triton到Ascend.md) | Triton → MLIR → AscendNPU-IR |

读完不需要记住全部细节，只要知道概念叫什么、在哪查即可。

---

## ③ 跑第一个 LLVM Pass

```bash
cd projects/hello-pass
chmod +x run.sh
./run.sh
```

预期能看到类似输出：

```text
Hello: add
  参数数量: 2
  基本块数量: 1
Hello: say_hello
  参数数量: 0
  基本块数量: 1
```

然后继续读：

- [LLVM IR 快速入门](./llvm/01-LLVM-IR快速入门.md)
- [第一个 LLVM Pass](./llvm/02-第一个LLVM-Pass.md)

---

## ④ 看一个 MLIR lowering demo

```bash
cd projects/ascendnpu-ir-demo
bash run-tests.sh
```

- 有 `mlir-opt`：执行 `.mlir` 文件中的 `// RUN:` 命令。
- 无 `mlir-opt`：自动切到标注检查模式，验证用例结构仍然完整。

推荐从这个文件开始看：

```text
projects/ascendnpu-ir-demo/test-cases/mlir/01_basic/01_vecadd.mlir
```

它展示了最小的 Linalg 向量加法，以及如何类比到 AscendNPU-IR 的 HFusion/HIVM lowering。

---

## 之后学什么？

| 阶段 | 内容 | 入口 |
|---|---|---|
| LLVM 进阶 | IR 变换、Pass 模式、调试测试 | [docs/llvm/README.md](./llvm/README.md) |
| MLIR | Dialect、Operation、Pattern、Lowering | [docs/mlir/README.md](./mlir/README.md) |
| Ascend 后端 | NPU 硬件、hfusion/hivm、构建调试 | [docs/ascend/README.md](./ascend/README.md) |
| 动手项目 | 所有示例项目索引 | [projects/README.md](../projects/README.md) |

## 遇到不认识的术语？

- 简短中英对照：[术语表](./glossary.md)
- 更详细解释：[技术术语速查手册](./reference/技术术语速查手册.md)
