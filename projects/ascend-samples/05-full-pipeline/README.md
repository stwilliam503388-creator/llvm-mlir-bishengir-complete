# 用例 5 — full-pipeline

一条 `linalg.generic` add 走过完整的 3 步 Lowering。

## Lowering 路径

```
linalg.generic { arith.addf }
     ↓ convert-linalg-to-hfusion
husion.elemwise_binary "add"
     ↓ convert-hfusion-to-hivm
hivm.load + hivm.vadd + hivm.store
     ↓ convert-hivm-to-llvm
llvm.load + llvm.fadd + llvm.store
```

## 运行

```bash
export ASCEND_BUILD=~/AscendNPU-IR/build
./trace.sh
```

## 回顾 5 个用例

| 用例 | 学了什么 |
|------|---------|
| 01 | linalg.generic 的结构 |
| 02 | 融合：30 行 → 5 行 |
| 03 | husion → hivm：融合 IR → 指令 |
| 04 | hivm → LLVM：回到熟悉的 LLVM |
| 05 | 完整路径：一条 add 走到底 |

**🎉 从 linalg 到 LLVM，你已经看完了一条 AI 算子完整的编译流程。**
