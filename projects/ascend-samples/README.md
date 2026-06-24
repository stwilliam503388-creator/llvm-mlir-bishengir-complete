# Ascend 编译器精选用例

从 AscendNPU-IR 源码中提取的 5 个关键用例，展示 linalg → hivm.hir → llvm 的完整 Lowering。

## 学习路径

```
用例 1: simple-add ──→ 看懂 linalg.generic → hivm.hir 的转变
用例 2: fusion-add-mul ──→ 融合：中间结果留在 Unified Buffer
用例 3: hivm 内部 ──→ alloc → load → compute → store 三步模式
用例 4: hivm-to-llvm ──→ 最终 Lowering 到通用 LLVM IR
用例 5: full-pipeline ──→ trace.sh 一键追踪全流程
```

## 用例列表

| # | 用例 | input | expected | 学什么 |
|---|------|-------|----------|--------|
| 01 | [simple-add](./01-simple-add/) | linalg.generic | hivm.hir.load+vadd+store | 语法对照 |
| 02 | [fusion-add-mul](./02-fusion-add-mul/) | linalg add+mul | hivm 共用 UB | 融合 |
| 03 | [hivm ops](./03-husion-to-hivm/) | linalg add | hivm 三步模式 | 内部结构 |
| 04 | [hivm-to-llvm](./04-hivm-to-llvm/) | hivm.hir | llvm.func | 最终 Lowering |
| 05 | [full-pipeline](./05-full-pipeline/) | linalg add | 完整追踪 | 全流程 |

每个用例含 `input.mlir` + `expected.mlir` + `README.md` 逐行解读。

## 运行

用例本身是阅读材料。要运行需要先构建 AscendNPU-IR：

```bash
cd AscendNPU-IR/build
./bin/bishengir-opt --pass-pipeline="convert-linalg-to-hivm" \
  ../../ascend-samples/01-simple-add/input.mlir
```

或用 `trace.sh` 一键追踪：
```bash
export ASCEND_BUILD=~/AscendNPU-IR/build
cd 05-full-pipeline && ./trace.sh
```

> 📖 构建指南 → [docs/ascend/03-构建与调试指南](../../docs/ascend/03-构建与调试指南.md)
