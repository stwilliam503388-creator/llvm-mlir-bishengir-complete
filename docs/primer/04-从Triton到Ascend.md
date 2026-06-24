# 04：从 Triton 到 Ascend——完整路径与你的定位

> 目标：了解大型 AI 编译器项目的常见文件结构
> 前置：[03-动手看MLIR长什么样](./03-动手看MLIR长什么样.md)
> 预估时间：10 分钟

> 阅读时间：5 分钟 | 前置知识：Primer 00-03

---

## 4.1 整条链路

从你写的 Triton Python 代码到最终在 Ascend NPU 上执行，编译器做的事情：

```text
你写的 Triton Python 代码        ← 你在这一层
  │ @triton.jit 触发编译
  ▼
TTIR（Triton 自己的 IR）         ← Triton 负责
  │ 转成 MLIR 标准格式
  ▼
linalg IR                       ← 这就是 test-cases/mlir/ 里的内容
  │ 这就是您现在学的内容          ← 👈 你在这里
  ▼
bishengir 的 HFusion IR         ← AscendNPU-IR 负责（融合优化）
  │ 
  ▼
bishengir 的 HIVM IR            ← AscendNPU-IR 负责（NPU 指令）
  │
  ▼
CANN SDK                        ← 华为负责，加载到 NPU
  │
  ▼
Ascend NPU 执行                  ← 最终结果
```

## 4.2 你需要掌握到什么程度

| 层级 | 你要做的事 | 需要懂多少 |
|------|-----------|-----------|
| **Triton Python** | 写 kernel | ✅ 熟练 |
| **TTIR** | 知道它在就行 | ⏺ 了解一下 |
| **linalg IR** | 能看懂，能调试 | ⭐ 重点掌握 ← 本项目焦点 |
| **HFusion/HIVM** | 知道概念即可 | ⏺ 了解一下 |
| **CANN** | 不需要懂 | ❌ 跳过 |

bishengir 需要你对 linalg IR 有扎实的理解。TTIR 到 linalg 的转换由 Triton 自动完成，bishengir 从 linalg 开始接管。

---

## 4.3 你现在在哪一步

```
Primer 00 → 01 → 02 → 03（实操）→ 04（当前）
                          ↓
                    docs/llvm/L00 → L01 → ...
                          ↓
                    docs/mlir/L00 → L01 → ...
                          ↓
                    回到 test-cases/ 亲手跑用例
                          ↓
                    读 references/ascendnpu-ir-mapping.md
```

**读完 Primer，你的下一步**：

| 步骤 | 做什么 | 时间 |
|------|--------|------|
| 1 | 读 `docs/llvm/L00-SSA.md`，深入理解 SSA | 30min |
| 2 | 跑 `projects/ascendnpu-ir-demo/variants/compare.sh` 看完整降级对比 | 10min |
| 3 | 打开 `docs/mlir/L00-MLIR概述.md` 了解 MLIR 设计理念 | 30min |
| 4 | 对照 `test-cases/triton/MAPPING.md`，看 MLIR ↔ Triton 双向映射 | 15min |

> ✅ **检查自己**：
> 1. Triton Python 代码到 Ascend NPU 执行，中间经过了几层 IR？
>    → 四层：TTIR → linalg → HFusion → HIVM。
> 2. 本项目重点关注的 IR 层是哪一层？
>    → linalg IR——bishengir 的输入层，也是调试时最常看到的层次。
