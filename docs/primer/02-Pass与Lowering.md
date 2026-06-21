# 003：Pass 与 Lowering（IR 是怎么变成机器码的）

> 阅读时间：8 分钟 | 前置知识：Primer 00, 01

---

## 3.1 Pass — IR 的"改造器"

假设你有一段 IR：

```llvm
%1 = add i32 %a, 0    ; a + 0
%2 = add i32 %1, %b   ; (a+0) + b
```

明眼人一看就知道 `%1 = add i32 %a, 0` 是多余的——加零等于没加。
编译器怎么做这种优化的？靠 **Pass**。

**Pass（遍 / 趟）** 的定义很简单：
**一个 Pass = 遍历一次 IR + 做一次修改。**

### 两种 Pass

```
分析 Pass (Analysis Pass):
  只看不改。
  遍历 IR，收集信息（用了多少变量、哪些变量没被使用），然后报告。
  类比：质检员到车间数一数今天产了多少件产品。

转换 Pass (Transform Pass):
  边看边改。
  遍历 IR，发现可以优化的地方，直接改掉。
  类比：质检员发现产品有毛刺，顺手打磨掉。
```

### 一个具体的例子

还是 `%1 = add i32 %a, 0`。转换 Pass 的处理流程：

```
Step 1: 遍历到 %1 = add i32 %a, 0
Step 2: 检查右操作数是否是 0
Step 3: 发现是 0，触发"加零消去"规则
Step 4: 把用到 %1 的地方全部替换成 %a
Step 5: 删除 %1 = add i32 %a, 0 这条指令

结果:
  优化前:  %1 = add i32 %a, 0; %2 = add i32 %1, %b
  优化后:  %2 = add i32 %a, %b     ← 少了一条指令
```

这个优化叫"常量折叠"——你会多次遇到。

### 在项目中找到它

打开 `projects/bishengir-op-counter/BishengirOpCounter.cpp`：

```cpp
// 这是一个分析 Pass：统计每种 op 出现了多少次
void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
        counters[op->getName().getStringRef()]++;
    });
    // 输出: func.func: 1, linalg.generic: 1, arith.addf: 1, ...
}
```

打开 `projects/bishengir-op-counter/BishengirPeelTranspose.cpp`：

```cpp
// 这是一个转换 Pass：把冗余 transpose 消除掉
// 检测 add(transpose(A), transpose(B)) → transpose(add(A,B))
LogicalResult matchAndRewrite(TransposeOp op, ...) {
    if (matchRedundantTranspose(op.getOperand()))
        return replaceOpWithNewOp<...>(...);
    return failure();
}
```

**理解这个区别很重要**：分析 Pass 是"只读"的，转换 Pass 是"写"的。

### Pass 管线

一个 Pass 通常只做一件事。要做多个优化，就把多个 Pass 串起来：

```
输入 IR
   ↓ Pass1: 常量折叠     %a + 0 → %a
   ↓ Pass2: 死代码消除   删除没用到的变量
   ↓ Pass3: 循环展开     小循环展开成直线代码
   ↓ Pass4: 向量化       标量 → 向量指令
输出 优化后的 IR
```

这就是编译器的"优化管线"。bishengir 的管线就是：

```
Linalg IR → HFusion → HIVM → NPU
    3个 Pass，串联执行
```

---

## 3.2 Lowering — 从高级到低级

Pass 能做优化，但编译器最终要解决另一个问题：**IR 太高层了，机器不认识。**

例如 `linalg.matmul` 在代码里只有一行：

```mlir
linalg.matmul ins(%A, %B) outs(%C)
```

但 CPU/NPU 不认识"矩阵乘法"这个概念——它只认识 load、add、mul、store。
所以需要把高级 IR **降低**成低级 IR。这个过程叫 **Lowering（降级）**。

### Lowering 的类比

```
"做一桌满汉全席"        ← linalg.matmul (1 行，高层语义)
       ↓
"先买菜 → 再洗菜 →      ← affine.for × 3 (18 行，中层)
 再切菜 → 再炒菜"
       ↓
"胡萝卜切 3mm 丁 →      ← LLVM IR (74 行，低层)
 热油至 180°C →         每条指令对应一个 CPU 操作
 翻炒 30 秒 → ..."
```

**为什么不能一步到位？**

因为中间层可以做**中间层优化**。

```text
例子: 先加再乘 → 可以融合
  C = A + B      ← 第一个 loop
  D = C * A      ← 第二个 loop
  如果降到 affine 再融合: 合并成一个 loop
  C[i] = A[i] + B[i]; D[i] = C[i] * A[i]
  → 内存少读一次，快 2 倍
```

这就是 MLIR 为什么要有多个 dialect（方言）：
- 在 Linalg 层做**算子融合**（矩阵 + 激活函数合并）
- 在 Affine 层做**循环优化**（分块、展开）
- 在 SCF 层做**控制流优化**（条件分支合并）
- 在 LLVM 层做**指令选择**（选最快的 CPU 指令）

**每层只做自己擅长的事。**

### 在项目中找到它

运行 `projects/bishengir-demo/variants/variant0_baseline.sh`：

```bash
bash projects/bishengir-demo/variants/compare.sh
```

你会看到 `matmul_4x4x4.mlir` 从 1 行变成 74 行的完整过程：

| 阶段 | IR 表示 | 行数 | 说明 |
|------|---------|------|------|
| 输入 | `linalg.matmul` | 1 行 | 高层语义：矩阵乘法 |
| 降级到 Affine | `affine.for × 3` | 18 行 | 展开成三重循环 |
| 降级到 LLVM | `llvm.load/add/mul/store` | 74 行 | 每条指令对应一个 CPU 操作 |

这就是编译器"从抽象到具体"的全过程。

---

## 3.3 Dialect — MLIR 为什么需要多个 IR

传统编译器只有一个 IR（比如 LLVM 的 IR）。
MLIR 的创新在于：**可以有多个 IR——每个叫一个 Dialect（方言）。**

| Dialect（方言） | 它描述的"方言词汇" | 好比 |
|----------------|-------------------|------|
| `toy` | add/mul/transpose/print | Toy 语言术语 |
| `linalg` | matmul/conv/generic | 线性代数术语 |
| `affine` | for/if/load/store | 循环和数组术语 |
| `scf` | for/while/if/yield | 结构化控制流术语 |
| `llvm` | add/load/call | CPU 指令术语 |
| **`hfusion`** | elemwise_binary/cube_matmul | **Ascend NPU 算子术语** |
| **`hivm`** | vadd/mload/mmul | **Ascend NPU 指令术语** |

### 为什么要有多个方言？

因为**一个问题拆成几层，每层独立解决，比一层硬扛简单得多**。

```
传统方案 (1 个 IR):
  矩阵乘 + 循环优化 + 代码生成 = 全部在一个 IR 里做
  → IR 变得巨大，加一个新功能要改整个框架

MLIR 方案 (N 个 IR):
  Linalg 层: 只操心矩阵乘语义
  Affine 层: 只操心循环优化
  LLVM 层: 只操心指令生成
  → 每层独立，加新硬件只需加新 dialect
```

**类比：团队的协作方式**

- 传统 1 个 IR = 一个人独立翻译整本书（质量依赖这个人水平）
- MLIR 多个 dialect = 三个人接力翻译（英→日→中），每段只翻自己最擅长的

### 在项目中找到它

`standalone-mlir` 项目定义了一个自定义 dialect：

```tablegen
// StandaloneOps.td — 定义 standalone 方言的 6 个操作
def AddOp : Standalone_Op<"add"> { ... }
def MulOp : Standalone_Op<"mul"> { ... }
def TransposeOp : Standalone_Op<"transpose"> { ... }
```

这就是一个最小的 dialect。`standalone-opt` 可以解析和打印这个 dialect 的 IR。

---

## 3.4 快速自测

1. **分析 Pass 和转换 Pass 的区别是什么？**
   - 答：分析 Pass 只看不改（统计 op 数量），转换 Pass 边看边改（消去冗余 transpose）

2. **Lowering 为什么不能一步到位？**
   - 答：中间层可以做中间层优化（比如循环融合），一步到位会丢失优化机会

3. **MLIR 为什么要用多个 dialect（方言）？**
   - 答：每层只做自己擅长的事——Linalg 层做矩阵优化，Affine 层做循环优化，互不干扰

4. **bishengir-demo 中 matmul 从 1 行变成 74 行，这是个问题吗？**
   - 答：不是问题，是 Lowering 的正常过程。bishengir 之所以能保持 1 行（hivm.mmul），是因为它有硬件支持——不需要展开到标量
