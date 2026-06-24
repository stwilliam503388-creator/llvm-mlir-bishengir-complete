# 快速入门

> 从零开始，一条路走到能写 LLVM Pass
> 总预估时间：2 小时
>
> 🤔 还不确定值不值得学？先花 5 分钟看 [为什么学 Ascend NPU 编译器？](./why-ascend.md)

---

## 路线图

```
① 环境检查 ──→ ② Primer 入门 ──→ ③ 写第一个 Pass
   (5 min)        (45 min)           (60 min)
```

---

## ① 环境检查（5 分钟）

```bash
# 在项目根目录运行
./setup.sh
```

看到 `✅ 所有核心依赖就绪` 即可进入下一步。

如果缺依赖 → [LLVM 环境搭建指南](./llvm/00-环境搭建.md)

---

## ② Primer 入门（45 分钟）

按顺序读，每篇 10-15 分钟：

| # | 文档 | 重点 | 时间 |
|---|------|------|------|
| 1 | [编译器是什么](./primer/00-编译器是什么.md) | 编译器 vs 解释器 | 10 min |
| 2 | [AST 与 IR](./primer/01-AST与IR.md) | 抽象语法树、中间表示 | 10 min |
| 3 | [Pass 与 Lowering](./primer/02-Pass与Lowering.md) | 编译器怎么优化 IR | 15 min |
| 4 | [从 Triton 到 Ascend](./primer/03-从Triton到Ascend.md) | AI 编译器的实际例子 | 10 min |

> 读完 primer 就够了 — 不需要全部记住，知道概念叫什么、在哪查就行。

---

## ③ 写第一个 Pass（60 分钟）

这是最有成就感的一步：

```bash
cd projects/hello-pass
chmod +x run.sh
./run.sh
```

你应该看到：

```
Hello: add
  参数数量: 2
  基本块数量: 1
Hello: say_hello
  参数数量: 0
  基本块数量: 1
```

**恭喜！你刚刚运行了人生第一个 LLVM Pass。**

然后跟着教程深入理解：

| # | 文档 | 内容 |
|---|------|------|
| 1 | [LLVM IR 快速入门](./llvm/01-LLVM-IR快速入门.md) | 读懂 IR 长什么样 |
| 2 | [第一个 LLVM Pass](./llvm/02-第一个LLVM-Pass.md) | 逐行解读 + 3 个挑战 |

完成 3 个挑战后，你就真正入门了编译器后端开发。

---

## 之后学什么？

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 2 进阶 | 更复杂的 LLVM Pass、IR 变换 | 计划中 |
| Phase 3 | [MLIR — 多层中间表示框架](./mlir/README.md) | ✅ 已完成 |
| Phase 4 | [Ascend NPU 编译器后端实战](./ascend/README.md) | ✅ 已完成 |

---

## 遇到不认识的术语？

→ [术语表](./glossary.md) — 中英对照，一句话解释
