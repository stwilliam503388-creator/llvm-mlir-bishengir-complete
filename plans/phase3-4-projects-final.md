# Phase 3 & 4 配套项目设计（最终版）

> 合并前两版，最终方案

---

## 最终交付：2 个项目 + 1 个精选集

| 项目 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `projects/mlir-hello/` | 新建，可运行 | 本项目 | MLIR 版 HelloPass |
| `projects/ascend-samples/` | 新建，精选集 | 从 AscendNPU-IR 挑选 | 5 个关键测试用例 + 导读 |

---

## 项目 1：`projects/mlir-hello/` — MLIR 版 HelloPass

同之前设计，对标 Phase 2 的 hello-pass。

---

## 项目 2：`projects/ascend-samples/` — Ascend 编译器精选用例

### 设计思路

从 AscendNPU-IR 的 131 个测试中只挑 **5 个**，每个 ≤50 行，串成一条完整的学习线：

```
用例 1: 最简单的逐元素 add（看懂 linalg 长什么样）
   ↓
用例 2: add+mul 融合（看懂融合做了什么）
   ↓
用例 3: husion → hivm Lowering（看懂从融合 IR 到指令）
   ↓
用例 4: hivm → LLVM Lowering（看懂最终代码生成）
   ↓
用例 5: 一条 add 走完全程（汇聚前 4 个用例，看完整变换）
```

### 目录结构

```
projects/ascend-samples/
├── README.md                    ← 总导读：5 个用例是什么关系
├── run-all.sh                   ← 一键运行全部 5 个用例（依赖 ascendnpu-ir 已构建）
│
├── 01-simple-add/
│   ├── input.mlir               ← linalg.generic 写的简单 add
│   ├── expected.mlir            ← 预期 Lowering 后的 husion IR
│   └── README.md                ← 逐行解读：这个 add 怎么变成 husion 的
│
├── 02-fusion-add-mul/
│   ├── input.mlir               ← 两个独立 linalg.generic（add + mul）
│   ├── expected.mlir            ← 融合后一个 husion.elemwise_binary
│   └── README.md                ← 解读：为什么 30 行变 5 行
│
├── 03-husion-to-hivm/
│   ├── input.mlir               ← husion.elemwise_binary "add"
│   ├── expected.mlir            ← hivm.vadd + hivm.load/store
│   └── README.md                ← 解读：指令级是什么样
│
├── 04-hivm-to-llvm/
│   ├── input.mlir               ← hivm.vadd + hivm.load/store
│   ├── expected.mlir            ← llvm.func + llvm.load/store
│   └── README.md                ← 解读：最后一步，回到 LLVM
│
└── 05-full-pipeline/
    ├── input.mlir               ← 同用例 1 的 linalg add
    ├── trace.sh                 ← 用 --mlir-print-ir-after-all 看每一步
    └── README.md                ← 汇总：从 linalg 到 llvm 的完整演变
```

### 每个用例的 README 格式（统一）

```markdown
# 用例 N：标题

> 对应文档：[docs/ascend/XX](../docs/ascend/XX)
> AscendNPU-IR 原始文件：bishengir/test/...

## 这段 IR 在干什么（一句话）

## 逐行解读

| 行 | 操作 | Dialect | 在干什么 |
|----|------|---------|---------|

## 运行

\`\`\`bash
cd AscendNPU-IR/build
./bin/ascendnpu-ir-opt --pass-pipeline="..." input.mlir
\`\`\`

## 预期输出（对照）

（左边 input.mlir，右边 expected.mlir）
```

### 用例 1 具体设计：`01-simple-add/input.mlir`

```mlir
// 最简单的逐元素加法
func.func @simple_add(%A: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<4xf32>, tensor<4xf32>)
    outs(%A : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
```

**expected.mlir**（经过 convert-linalg-to-hfusion 后）：

```mlir
func.func @simple_add(%A: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
  %0 = husion.elemwise_binary "add" ins(%A, %B) outs(%A)
      : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
```

**关键教学点**：14 行 linalg → 3 行 husion。学习者能直观看到：编译器把"复杂描述"翻译成了"简单操作"。

### 用例 5 的 `trace.sh`

```bash
#!/bin/bash
# 追踪一条 add 指令经过所有 Pass 的完整变换

PIPELINE="builtin.module(
  convert-linalg-to-hfusion,
  convert-hfusion-to-hivm,
  convert-hivm-to-llvm
)"

echo "=== 原始 linalg IR ==="
cat input.mlir

echo ""
echo "=== Step 1: linalg → husion ==="
mlir-opt --pass-pipeline="$PIPELINE" input.mlir \
  --mlir-print-ir-after-all 2>&1 | head -80
```

### 用例与文档的对应关系

| 用例 | 对应文档 | 学什么 |
|------|---------|--------|
| 01-simple-add | `ascend/01-husion-hivm-Dialect详解.md` | linalg.generic 的结构 |
| 02-fusion-add-mul | `ascend/01` 融合节 | 为什么融合减少数据搬运 |
| 03-husion-to-hivm | `ascend/01` hivm 节 | 从融合 IR 到虚拟指令 |
| 04-hivm-to-llvm | `ascend/01` 末尾 | 最终回到 LLVM IR |
| 05-full-pipeline | `ascend/02-一个Ascend-Pass详解.md` | 完整 Lowering 追踪 |

### 依赖说明

```
projects/ascend-samples/ 本身是"阅读材料"。
运行用例需要先构建 AscendNPU-IR：

  git clone https://github.com/Ascend/AscendNPU-IR.git
  cd AscendNPU-IR && mkdir build && cd build
  cmake .. && make -j
```

---

## 最终交付清单

| # | 项目 | 文件数 | 定位 |
|---|------|--------|------|
| 1 | `projects/mlir-hello/` | 5 | Phase 3 动手入口（对标 hello-pass） |
| 2 | `projects/ascend-samples/` | 16 (5×3 + README + run-all.sh) | Phase 4 精选用例（从 AscendNPU-IR 131 个测试中挑 5 个） |

**总计**：新建 2 个项目，21 个文件。约 600 行代码 + 注释。

### 和 hello-pass 的关系

```
Phase 2: hello-pass       → 一键运行 LLVM Pass       → 建立信心
Phase 3: mlir-hello       → 一键运行 MLIR Pass       → 感受 MLIR 和 LLVM 的异同
Phase 4: ascend-samples   → 5 个用例串起完整 Lowering → 理解真实工业项目
```
