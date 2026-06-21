---
type: plan
status: executed
project: llvm-mlir-bishengir-complete
created: 2026-06-21
target: AscendNPU-IR 对接方案
---

> ✅ **此计划已执行完毕**。结果见 `test-cases/mlir/`、`docs/primer/`、`references/ascendnpu-ir-mapping.md`、`docs/ascendnpu-ir/` 等。

# AscendNPU-IR 对接方案

## 一、当前缺陷分析

### 缺陷 1：只有概念级描述，没有源码级对应

| 现有内容 | 缺陷 |
|---------|------|
| README 写着 "bishengir 三阶段: Linalg→HFusion→HIVM" | 但没说 **具体哪个文件** 实现了哪个阶段 |
| references/README.md 写着 "核心目录: bishengir/lib/Conversion/" | 但没说 **该目录下每个 Pass 分别做什么** |
| bishengir-demo 在跑 `--convert-linalg-to-affine-loops` | 但没说 **这个 pass 在 ascendnpu-ir 源码里对应哪个 .cpp 文件** |
| bishengir-op-counter 在写自定义 Pass | 但没说 **如果你想把它注册到 bishengir-opt，应该怎么操作** |

**效果**：读者知道了"有这回事"，但拿到 AscendNPU-IR 源码后还是不知道从哪看起。

### 缺陷 2：已有的翻译文档和分析笔记没整合进来

| 已有资料（在 Obsidian vault） | 大小 | 状态 |
|------|------|------|
| 9 篇翻译文档（Pass + Dialect） | ~112KB | ❌ 不在 repo 中 |
| BishengIR代码仓库解读.md | 21.5KB | ❌ 不在 repo 中 |
| AscendNPUIR文档总结.md | 32KB | ❌ 不在 repo 中 |
| 本地 ascendnpu-ir docs 目录 | 多篇 | ❌ 不在 repo 中 |

**效果**：读者要理解 AscendNPU-IR 的 AnnotationPass / HACCPass 等细节，需要自己去翻官方文档。

### 缺陷 3：没有源码阅读路径

没有人告诉读者：拿到 AscendNPU-IR 源码后，**先看哪个文件、再看哪个文件、最后看哪个文件**。
没有把本项目的 Stage 0→1→2→3 学习路径映射到 AscendNPU-IR 的目录结构上。

### 缺陷 4：代码本身没有 AscendNPU-IR 标注

bishengir-demo 的 test-cases（vecadd/matmul/fused）只是纯 MLIR 文件，没有注释说明：
- 这个 `linalg.generic` 在 ascendnpu-ir 的哪个测试用例里也有？
- 这个 `arith.addf` 在 bishengir 里被转换成了什么？
- 这个降级过程和 ascendnpu-ir 的 LinalgToHFusion 有什么区别？

---

## 二、新增方案

### 方案 A：文档层 — `references/ascendnpu-ir-mapping.md`

新增一份完整的 **三栏对应对照文档**，一次性修复缺陷 1 + 3。

| 章 | 内容 | 解决缺陷 |
|----|------|---------|
| **1. 三阶段降级源码追踪** | 每个 Pass 对应的 ascendnpu-ir 源码路径 + 核心函数 + 输入输出格式 | 缺陷 1 |
| **2. Dialect 源码追踪** | 每个 dialect（hfusion/hivm）对应的 .td 文件 + .cpp 实现 | 缺陷 1 |
| **3. 翻译文档索引** | 9 篇翻译文档的标题、原始文件路径、在本项目中的位置 | 缺陷 2 |
| **4. 深度分析笔记索引** | BishengIR解读 + 文档总结 的位置和内容概要 | 缺陷 2 |
| **5. 阅读路径** | 拿到源码后，先看 A → 再看 B → 最后看 C 的顺序 | 缺陷 3 |
| **6. 项目工程源码追踪** | 本项目的每个工程文件 → 对应 ascendnpu-ir 的哪个目录 | 缺陷 4 |

#### 6.1 节：项目工程源码追踪（关键内容）

| 本项目文件 | 对应 ascendnpu-ir 源码 |
|-----------|----------------------|
| `projects/bishengir-demo/test-cases/vecadd_128.mlir` | `bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir` |
| `projects/bishengir-demo/test-cases/matmul_4x4x4.mlir` | `bishengir/test/Conversion/LinalgToHFusion/matmul-to-hfusion.mlir` |
| `projects/bishengir-demo/variants/variant0_baseline.sh` | `bishengir-opt --convert-linalg-to-hfusion` |
| `projects/bishengir-demo/variants/variant3_hw_mapping.sh` | `bishengir/lib/Conversion/HFusionToHIVM/hfusion-to-hivm.cpp` |
| `projects/bishengir-op-counter/BishengirOpCounter.cpp` | `bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp`（模式） |
| `projects/bishengir-op-counter/BishengirPeelTranspose.cpp` | `bishengir/lib/Conversion/ToyCombine.cpp`（模式） |
| `projects/standalone-mlir/include/standalone/StandaloneOps.td` | `bishengir/include/bishengir/Dialect/HFusion/HFusionOps.td` |
| `projects/standalone-mlir/tools/standalone-opt.cpp` | `bishengir/tools/bishengir-opt/bishengir-opt.cpp` |

#### 5 节：阅读路径（关键内容）

```text
拿到 AscendNPU-IR 源码后，按这个顺序读：

Step 1: 读 bishengir-opt 入口
  → tools/bishengir-opt/bishengir-opt.cpp
  → 看 Pass 是怎么注册的，dialect 是怎么加载的

Step 2: 读一个简单的 Pass
  → lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
  → 看 ConversionTarget 怎么设，RewritePattern 怎么写

Step 3: 读 Dialect 定义
  → include/bishengir/Dialect/HFusion/HFusionOps.td
  → 看 Op 是怎么用 TableGen 定义的

Step 4: 读测试用例
  → test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
  → 看输入 IR 和输出 IR 的格式

Step 5: 对照本项目
  → docs/llvm/ 和 docs/mlir/ 对应这里的概念
  → projects/ 对应这里的代码模式
```

---

### 方案 B：文档层 — 翻译文档 + 分析笔记收录

将 Obsidian vault 中的 11 篇已有资料复制到 `docs/ascendnpu-ir/` 目录下：

```
docs/ascendnpu-ir/
├── README.md                  ← 翻译文档索引 + 阅读说明
├── translations/               ← 9 篇翻译文档
│   ├── 01-AnnotationPasses.md
│   ├── 02-HACCPasses.md
│   ├── 03-ScopePasses.md
│   ├── 04-SymbolPasses.md
│   ├── 05-AnnotationDialect.md
│   ├── 06-ScopeDialect.md
│   ├── 07-SymbolDialect.md
│   ├── 08-MathExtDialect.md
│   └── 09-MemRefExtDialect.md
├── analysis/
│   ├── BishengIR代码仓库解读.md       — 仓库结构逐目录解读
│   └── AscendNPUIR文档总结.md         — 文档体系总结
└── code-mapping.md           ← 方案 A 的代码追踪对照
```

---

### 方案 C：代码层 — 给工程加 AscendNPU-IR 注释

#### C1: bishengir-demo test-cases 加头部注释

每个 `.mlir` 文件头部新增注释块，标明对应的 ascendnpu-ir 源码路径：

```mlir
// 对应 ascendnpu-ir:
//   ├── 输入格式: bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
//   ├── Pass 实现: lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
//   └── 预期输出: hfusion.elemwise_binary {fun = add}
```

#### C2: bishengir-op-counter 源码加文件级注释

```cpp
// BishengirOpCounter.cpp
// 分析 Pass，统计 ops 分布。
// 参考 ascendnpu-ir:
//   - 模式: lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp (walk 遍历)
//   - 测试: test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
//
// 如果要注册到 bishengir-opt:
//   1. 复制到 bishengir/lib/Conversion/BishengirOpCounter/
//   2. 在 CMakeLists.txt 添加 add_mlir_conversion_library
//   3. 在 InitAllPasses.h 添加注册
```

#### C3: run-demo.sh 和 variant 脚本加对照注释

在每个脚本头部标注对应的 ascendnpu-ir 命令行：

```bash
# 本命令等价于:
#   bishengir-opt --convert-linalg-to-hfusion --convert-arith-to-hfusion \
#                 --convert-hfusion-to-hivm vecadd.mlir
# 区别: bishengir 用 hivm.vadd（向量指令），本 demo 用 llvm.add（标量指令）
```

---

### 方案 D：对接层 — `scripts/` 目录下加实用脚本

新增工具脚本，帮助读者在 ascendnpu-ir 源码里定位：

| 脚本 | 功能 |
|------|------|
| `scripts/trace-to-ascendnpu.sh` | 给定一个 MLIR pass 名，输出它对应 ascendnpu-ir 的哪个文件 |
| `scripts/grep-ascendnpu.sh` | 在 ascendnpu-ir 源码中搜索关键词（需本地有源码） |

---

## 三、方案优势总结

| 缺陷 | 补丁 | 效果 |
|------|------|------|
| 缺陷 1：没有源码级对应 | 方案 A（mapping 文档）+ 方案 C（代码注释） | 每个 pass / dialect / test-case 都能直接定位到 ascendnpu-ir 源码文件 |
| 缺陷 2：翻译文档未收录 | 方案 B（复制 11 篇到 repo） | 读者无需翻 Obsidian，在 repo 内就能看中文翻译 |
| 缺陷 3：没有阅读路径 | 方案 A §5（5 步阅读顺序） | 拿到源码第一分钟就知道看哪 |
| 缺陷 4：代码没有 AscendNPU-IR 标注 | 方案 C（文件头部注释）| 每个 `.mlir` 文件和 `.cpp` 文件都标注了对应关系 |
| 新增：需要实用工具 | 方案 D（grep/trace 脚本） | 在终端快速定位 |

---

## 四、实施步骤

```
Step 1: 写 references/ascendnpu-ir-mapping.md    (≈ 300 行)
Step 2: 建 docs/ascendnpu-ir/，复制 11 篇已有资料   (≈ 160KB)
Step 3: 给 bishengir-demo test-cases 加头部注释    (3 个 .mlir 文件)
Step 4: 给 bishengir-op-counter 加文件级注释       (2 个 .cpp 文件)
Step 5: 给 variant 脚本加对照注释                    (4 + 1 个 .sh 文件)
Step 6: 创建 scripts/trace-to-ascendnpu.sh          (1 个 .sh 文件)
Step 7: 更新 references/README.md 汇总信息
Step 8: 更新 README.md 项目总览 + 资源章节
Step 9: git add + commit + push
```

### 工作量预估

| 步骤 | 预计 | 并行性 |
|------|------|--------|
| Step 1: mapping 文档 | ~15 分钟 | 独立 |
| Step 2: 复制 11 篇 | ~5 分钟（脚本批量） | 独立 |
| Step 3: test-case 注释 | ~10 分钟（3 个文件） | 与 Step 4 可并行 |
| Step 4: op-counter 注释 | ~5 分钟（2 个文件） | 与 Step 3 可并行 |
| Step 5: variant 脚本注释 | ~10 分钟（5 个脚本） | 与 Step 3-4 可并行 |
| Step 6: 实用脚本 | ~5 分钟 | 独立 |
| Step 7-8: README 更新 | ~5 分钟 | 最后 |
| **合计** | **~50 分钟** | |

---

## 五、不做的范围

以下不在本次方案中：

- ❌ ascendnpu-ir 源码的全量编译（需要 init 子模块 + 1h+ 编译）
- ❌ 将 bishengir-op-counter 实际注册到 bishengir-opt（需要编译环境）
- ❌ 9 篇翻译文档的审校（已翻译，直接收录）
- ❌ 新增对 AscendNPU-IR 其他 dialect 的翻译（Annotation/Scope/Symbol 等已有，不做新翻译）
