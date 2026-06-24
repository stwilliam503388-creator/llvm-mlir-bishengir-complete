# 用例 5 — full-pipeline

一条 `linalg.generic` add 走过 AscendNPU-IR 的完整 Lowering。

## Lowering 路径

```
linalg.generic { arith.addf }
     ↓ convert-linalg-to-hivm
hivm.hir.load + hivm.hir.vadd + hivm.hir.store
     ↓ convert-hivm-to-llvm
llvm.load + llvm.fadd + llvm.store
```

## 运行

```bash
export ASCEND_BUILD=~/AscendNPU-IR/build
chmod +x trace.sh && ./trace.sh
```

## 回顾 5 个用例

| 用例 | 学了什么 |
|------|---------|
| 01 | linalg.generic 和 hivm.hir 的语法对照 |
| 02 | 融合：中间结果留在 Unified Buffer，不写回 HBM |
| 03 | hivm 三步模式：alloc→load→compute→store |
| 04 | hivm→LLVM：最终回到通用 IR |
| 05 | 完整路径：一条 add 从 linalg 到 llvm |

**🎉 你已经看完了一条 AI 算子在 Ascend NPU 上的完整编译流程。**
