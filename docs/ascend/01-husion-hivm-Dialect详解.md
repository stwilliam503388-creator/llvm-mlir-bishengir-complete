# 01 — husion 和 hivm Dialect 详解

> 目标：理解两个核心 Dialect 的完整 Lowering 流程
> 前置：[00 — Ascend NPU 硬件概述](./00-Ascend-NPU硬件概述.md)
> 预估时间：20 分钟

## 1. 为什么需要两个 Dialect？

| 阶段 | Dialect | 对应 | 关注什么 |
|------|---------|------|---------|
| 高级 | `linalg` | 菜谱 | 做什么菜 |
| 中级 | `husion` | 备菜计划 | 怎么优化流程 |
| 低级 | `hivm` | 厨房操作 | 具体怎么做 |

每层只管一层的事。

## 2. husion — 昇腾融合 IR

**核心目标**：多个独立操作**融合**成一个，减少数据搬运。

### 为什么融合重要？

```
没有融合：
  数据 HBM→L1→乘法→写回 HBM
  数据 HBM→L1→加法→写回 HBM
  共 4 次搬运

融合后：
  数据 HBM→L1→乘法→加法→写回 HBM
  共 2 次搬运，节省 50%
```

### 核心操作

```mlir
husion.elemwise_binary "add" ins(%a, %b) outs(%c)     // 逐元素 add
husion.matmul ins(%a, %b) outs(%c)                    // 矩阵乘法
husion.relu ins(%a) outs(%b)                          // 激活函数
```

### 融合示例

```mlir
// 融合前：两个独立 linalg.generic
linalg.generic ... { ^bb0: arith.addf ... }   // add
linalg.generic ... { ^bb0: arith.mulf ... }   // mul
         ↓ ConvertLinalgToHusion
// 融合后：一个 husion
husion.elemwise_binary "add_mul" ins(%A, %B, %C) outs(%A)
```

**30 行 → 5 行，10 倍压缩。**

## 3. hivm — 昇腾虚拟指令

**核心目标**：描述 NPU 能执行的基本操作。

### 核心操作

```mlir
hivm.vadd %a, %b : vector<256xf32>     // 向量加法（Vector Unit）
hivm.vmul %a, %b : vector<256xf32>     // 向量乘法
hivm.mmad %a, %b : vector<16x16xf16>   // 矩阵乘法（Cube Unit）
hivm.load %addr : memref<1024xf32>     // 从 HBM 加载
hivm.store %v, %addr                   // 存回 HBM
hivm.barrier                           // 同步等待
```

### Lowering 示例

```mlir
// husion → hivm
husion.elemwise_binary "add" ins(%A, %B) outs(%C)
         ↓ ConvertHusionToHIVM
%vA = hivm.load %addrA : vector<256xf32>
%vB = hivm.load %addrB : vector<256xf32>
%vC = hivm.vadd %vA, %vB : vector<256xf32>
hivm.store %vC, %addrC
```

## 4. 完整 Lowering 路径

```
linalg.generic (框架看到的)
     │  ConvertLinalgToHusion
     ▼
husion.elemwise_binary (融合优化)
     │  ConvertHusionToHIVM
     ▼
hivm.vadd + hivm.load/store (接近硬件)
     │  代码生成
     ▼
Ascend NPU 可执行文件
```

## 验证

- [ ] 能说出 husion 解决什么问题（融合）
- [ ] 能说出 hivm 解决什么问题（指令描述）
- [ ] 能画出 linalg → husion → hivm 路径
- [ ] 知道 husion 和 hivm 的区别

> 📖 [术语表](../glossary.md)
> **下一步**：[02 — 一个 Ascend Pass 详解](./02-一个Ascend-Pass详解.md)
