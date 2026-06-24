# 02：Pass 和 Lowering（IR 是怎么变成机器码的）

> 目标：理解 Pass 和 Lowering 的概念
> 前置：[01-AST与IR](./01-AST与IR.md)
> 预估时间：15 分钟

> 目标：理解 Pass 和 Lowering 的概念
> 前置：[01-AST与IR](./01-AST与IR.md)
> 预估时间：15 分钟


> 阅读时间：6 分钟 | 前置知识：Primer 00, 01
> 遇到不认识的术语 → 查 `docs/reference/技术术语速查手册.md`

---

## 3.1 IR 不是一步到位的

从你写的代码到最终机器码，IR 要经过**多次转换**：

```text
你写的:      C = A + B

linalg IR:   linalg.generic { arith.addf }     ← 高级（像"炒菜"）
    ↓
affine IR:   affine.for 循环                    ← 中级（像"热锅→倒油→炒"）
    ↓
scf/cf IR:  scf.for / cf.br                    ← 中低级（循环展开成判断和跳转）
    ↓
LLVM IR:    %1 = load; %2 = add; store          ← 低级（接近机器指令）
```

每一层 IR 都比上一层更"啰嗦"、更接近硬件。这个过程叫 **Lowering（降级）**。

---

## 3.2 Lowering——从高层语义到底层指令

**Lowering** = 把"高级但难执行"的 IR 变成"低级但易执行"的 IR。

```
linalg.generic "把 A 和 B 加起来"     ← 一句话说清楚，但 CPU 不知道怎么执行
    ↓ lowering
affine.for + arith.addf               ← 一步步说清楚：先读 A[0]，加 B[0]，存到 C[0]...
    ↓ lowering
load + add + store                     ← 最啰嗦，但 CPU 可以直接执行
```

就像做菜：
- **高层**（linalg）："炒个番茄炒蛋" —— 一句话，意思清楚，但没告诉你怎么做
- **中层**（affine）："打蛋 → 切番茄 → 热锅 → 倒油 → 炒蛋 → 盛出 → 炒番茄 → 混合"
- **低层**（LLVM）：每一步精确到秒和温度

**多层 IR 的好处**：每层可以做该层特有的优化。比如 linalg 层可以做"把两个操作融合成一个"（高层优化），affine 层可以做"循环展开"（中层优化）。

> 💡 **Dialect（方言）**是 MLIR 里"不同层的 IR"。每个 dialect 有自己特有的操作——linalg 有 `linalg.matmul`（矩阵乘），affine 有 `affine.for`（循环），arith 有 `arith.addf`（加法）。
> 在本项目中：`linalg` → `affine` → `scf` → `llvm` 就是一条完整的 lowering 链。
> 你现在只需要知道"有很多层"就够了。具体 dialect 的定义是进阶内容。

---

## 3.3 Pass——IR 的"改造器"

**Pass（遍）** = 遍历一次 IR，做一次修改或检查。

有两种 Pass：

| 类型 | 做什么 | 类比 | 本项目的例子 |
|------|--------|------|------------|
| **分析 Pass** | 只看不改，统计信息 | 质检员数产品数量 | `BishengirOpCounter.cpp` |
| **转换 Pass** | 边看边改，做优化 | 质检员顺手修毛刺 | `BishengirPeelTranspose.cpp` |

一条 lowering 流水线就是**一连串 Pass**：

```text
mlir-opt --pass1 --pass2 --pass3 input.mlir
           ↓       ↓       ↓
        转换 A → 转换 B → 转换 C → 最终 IR
```

---

## 3.4 回到 bishengir

bishengir（AscendNPU-IR）的 lowering 流水线：

```text
Triton IR      →     linalg IR      →    hfusion IR     →     hivm IR
                        ↓                   ↓                    ↓
                   LinalgToHFusion     ArithToHFusion     HFusionToHIVM
                   (本项目的基准)       (融合优化)          (NPU 指令生成)
```

这就是为什么本项目研究 `linalg → affine → llvm` 这条标准路径——因为 bishengir 的入口就是 linalg IR。理解了这条路径，你就理解了 bishengir 的一半。

> ✅ **检查自己**：
> 1. Lowering 是什么？
>    → 从高级 IR 到低级 IR 的转换过程，每一步都更啰嗦但更容易执行。
> 2. 为什么不用一层 IR 搞定？
>    → 不同层可以针对性地做不同优化（高层做融合，中层做循环优化）。
> 3. 分析 Pass 和转换 Pass 的区别？
>    → 分析只看不改，转换边看边改。
