# Ascend 编译器精选用例

从 AscendNPU-IR 的 131 个测试中精选 5 个，串成完整学习线。

## 学习路径

```
用例 1: simple-add ──→ 看懂一个 linalg.generic
      ↓                  (14行 linalg → 3行 husion)
用例 2: fusion-add-mul ──→ 两个操作融合为一个
      ↓                  (30行 → 5行)
用例 3: husion-to-hivm ──→ 融合 IR → 虚拟指令
      ↓
用例 4: hivm-to-llvm ──→ 虚拟指令 → LLVM IR
      ↓
用例 5: full-pipeline ──→ 一条 add 走完全程
```

## 用例列表

| # | 用例 | 学什么 |
|---|------|--------|
| 01 | [simple-add](./01-simple-add/) | linalg.generic 的结构 |
| 02 | [fusion-add-mul](./02-fusion-add-mul/) | 为什么融合减少数据搬运 |
| 03 | [husion-to-hivm](./03-husion-to-hivm/) | 从融合 IR 到虚拟指令 |
| 04 | [hivm-to-llvm](./04-hivm-to-llvm/) | 最终回到 LLVM IR |
| 05 | [full-pipeline](./05-full-pipeline/) | 完整 Lowering 追踪 |

## 使用方式

用例本身是阅读材料。运行需要先构建 AscendNPU-IR：

```bash
cd AscendNPU-IR/build
./bin/ascendnpu-ir-opt --pass-pipeline="..." ../../ascend-samples/01-simple-add/input.mlir
```

> 📖 构建指南 → [docs/ascend/03-构建与调试指南](../../docs/ascend/03-构建与调试指南.md)
