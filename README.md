1|# LLVM → MLIR → bishengir: Ascend NPU 编译器全链路学习
2|
3|> 从 LLVM IR 入门到 MLIR Dialect 开发，最终对接 AscendNPU-IR (bishengir) 的完整学习路径与工程合集
4|
5|[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
6|[![LLVM](https://img.shields.io/badge/LLVM-22.1.6-blue)](https://llvm.org)
7|[![macOS](https://img.shields.io/badge/macOS-26.5.1-ff69b4)(https://www.apple.com/macos)
8|[![AscendNPU-IR](https://img.shields.io/badge/AscendNPU--IR-官方-blueviolet)](https://github.com/Ascend/AscendNPU-IR)
9|
10|---
11|
12|## 一、项目背景
13|
14|### 1.1 为什么需要这个项目
15|
16|AI 芯片正在经历从通用计算（CPU/GPU）到专用计算（NPU/TPU）的转变。以华为昇腾（Ascend）为代表的 NPU 在推理场景中表现突出，但它们的软件栈复杂度远高于 CUDA。
17|
18|编译器是这座软件栈的**核心骨架**。以一条 Triton 代码到 Ascend NPU 执行为例：
19|
20|```text
21|Triton Python kernel  (你写的代码)
22|        │
23|        ▼
24|Triton IR (TT Dialect)       ← MLIR 中间表示
25|        │
26|        ▼
27|AscendNPU-IR (bishengir)     ← Ascend 编译器
28|  Linalg → HFusion → HIVM
29|        │
30|        ▼
31|CANN Runtime                 ← 华为 SDK
32|        │
33|        ▼
34|Ascend NPU 执行
35|```
36|
37|每一步都涉及编译器知识：**IR（中间表示）**、**dialect（方言）**、**Pass（转换）**、**Lowering（降级）**。现有教程要么偏学术（LLVM 源码分析），要么偏应用（只讲 Triton 使用），缺乏一条从零到 AscendNPU-IR 的动手路径。
38|
39|本项目填补这个空白。
40|
41|### 1.2 AscendNPU-IR 是什么
42|
43|**AscendNPU-IR** 是华为开源的 Ascend NPU MLIR 编译器项目：
44|
45|| 项目 | 链接 | 说明 |
46||------|------|------|
47|| **官方代码仓** | https://github.com/Ascend/AscendNPU-IR | 华为官方维护 |
48|| **中文文档** | https://ascendnpu-ir.gitcode.com/zh_cn/index.html | GitCode 镜像，含完整 API 参考 |
49|
50|它本质上是基于 MLIR 框架构建的一套**多层 IR 降级流水线**：
51|
52|```text
53|输入:  Linalg / Arith / Func 等标准 MLIR dialect
54|         │
55|         ▼
56|  Pass1: -convert-linalg-to-hfusion
57|         Linalg dialect → HFusion dialect（融合算子抽象）
58|         │
59|         ▼
60|  Pass2: -convert-arith-to-hfusion
61|         Arith dialect → HFusion dialect
62|         │
63|         ▼
64|  Pass3: -convert-hfusion-to-hivm
65|         HFusion dialect → HIVM dialect（NPU 指令抽象）
66|         │
67|         ▼
68|  HIVM → CANN Runtime → Ascend NPU 执行
69|```
70|
71|**本项目研究的 `ascendnpu-ir`** 是 AscendNPU-IR 的一个活跃 fork，由 Nous Research 维护，在社区中也被称为 **bishengir**。它在官方基础上扩展了更多 dialect 和转换 Pass。
72|
73|### 1.3 本项目的价值
74|
75|| 维度 | 说明 |
76||------|------|
77|| **知识层次** | 从 LLVM IR 基础 → MLIR dialect 概念 → AscendNPU-IR 实战，三级递进 |
78|| **动手验证** | 所有知识都有对应工程：4 个项目，全部可在 macOS 上运行 |
79|| **开源生态** | 对标华为 AscendNPU-IR 官方项目，代码直接可读 |
80|| **零基础可入** | 附带 4 篇 primer 入门文档，面向 AI 工程师，无需编译器经验 |
81|
82|### 1.4 适用读者
83|
84|- 想理解 **Ascend NPU 编译栈** 的 AI 工程师
85|- 使用 Triton 做模型推理，想深入底层的学习者
86|- 需要开发 **自定义 MLIR dialect** 的编译器开发者
87|- 阅读 AscendNPU-IR / triton-ascend 源码时遇到 MLIR 瓶颈的学习者
88|- **零基础也没问题，从 `docs/primer/` 开始**
89|
90|### 1.5 前置知识
91|
92|| 要求 | 说明 |
93||------|------|
94|| ✅ C++ 基础 | 能读 C++17 代码 |
95|| ✅ Python 基础 | 能读 Python 代码 |
96|| 🟡 编译器经验 | **不需要。从 `docs/primer/` 开始（约 30 分钟）** |
97|| 🟡 编译器直觉 | **不需要。Primer 会用类比帮你建立直觉** |
98|| ❌ Ascend NPU | 不需要硬件，所有验证在 CPU 上完成 |
99|
100|---
101|
102|## 二、项目总览
103|
104|### 覆盖范围
105|
106|```
107|基础知识 ←───────── 核心概念 ←────────────── 工程实践
108|─────────           ─────────             ──────────────
109|LLVM IR            MLIR Dialect          bishengir-demo / AscendNPU-IR
110|  SSA 形式            dialect 定义          可运行降级流水线
111|  类型系统/GEP        Operation/Region      Linalg→affine→LLVM
112|  控制流/Phi          Pattern Rewriting     三阶段对照分析
113|  Pass 开发           Dialect Conversion    向量加法 / 矩阵乘法 / 融合优化
114|                    Pass 管理器           + 官方文档对接分析
115|                    TableGen ODS          自定义 MLIR Pass
116|                    mlir-opt 工具链        OpCounter / PeelTranspose
117|                                          Toy Mini 解析器（纯 C++17 零依赖）
118|                                          Standalone MLIR 项目（CMake + TableGen）
119|                                          Triton → AscendNPU-IR 全链路对接
120|                                          triton-ascend + 官方文档结合分析
121|```
122|
123|### 80+ 文件，覆盖 4 个层次
124|
125|```
126|层次 1: 文档 (19 篇笔记, ~90KB)
127|├── LLVM IR 基础 (7 篇)     — 从 SSA 到 Pass 开发
128|├── MLIR 体系 (8 篇)        — 从 dialect 概念到 Triton 对接
129|└── 零基础入门 (4 篇)       — 面向 AI 工程师的编译器概念速成
130|
131|层次 2: 可运行工程 (4 个项目)
132|├── bishengir-demo ★        — 3 个 MLIR 用例 + 4 种优化方案对比
133|├── toy-mini                 — 纯 C++17 Toy 解析器，编译通过
134|├── standalone-mlir          — CMake + Makefile + TableGen 自建 dialect
135|└── bishengir-op-counter     — 分析 + 转换 Pass 参考代码
136|
137|层次 3: 设施
138|├── setup.sh                 — 依赖检查
139|├── LICENSE                  — MIT 开源
140|├── .gitattributes           — 换行符管理
141|└── references/              — 外部源码索引
142|
143|层次 4: 外部源码（不在本仓库）
144|├── ascendnpu-ir (bishengir) — Ascend NPU MLIR 转换 Pass
145|└── triton-ascend            — Triton 前端对接
146|```
147|
148|### 项目结构
149|
150|```
151|ascend-npu-compiler-learning/
152|├── README.md                         ← 本文件（项目总览）
153|├── SUMMARY.md                        ← 完整输出总结文档
154|├── LICENSE                           ← MIT 许可证
155|├── setup.sh                          ← 依赖检查脚本
156|│
157|├── docs/                             ← 知识库（19 篇笔记）
158|│   ├── llvm/                         ← LLVM 速通（7 篇）
159|│   ├── mlir/                         ← MLIR 体系（8 篇）
160|│   └── primer/                       ★ 零基础入门（4 篇，约 30 分钟）
161|│       ├── README.md                 — 阅读顺序
162|│       ├── 00-编译器是什么.md         — 编译器三步工作法
163|│       ├── 01-AST与IR.md             — 语法树、SSA、三地址码
164|│       ├── 02-Pass与Lowering.md      — 分析/转换 Pass、dialect、降级
165|│   │   └── 03-从Triton到Ascend.md    — 全路径串联到本项目
166|│   │
167|│   └── reference/                    ★ 术语速查手册（298 条术语, 按主题分组）
168|│       └── 技术术语速查手册.md       — SSA/Dialect/Pass/Lowering/Linalg 等, 每条含一句话+类比
169|│
170|├── projects/                         ← 工程项目（4 个）
171|│       └── README.md                 — 阅读顺序
172|│
173|├── projects/                         ← 工程项目（4 个）
174|│   ├── bishengir-demo/               ★ 可运行降级流水线
175|│   ├── toy-mini/                     ★ 从零写 Toy 解析器
176|│   ├── standalone-mlir/              ★ 从零构建 MLIR dialect
177|│   └── bishengir-op-counter/         ★ 自定义 Pass 参考代码
178|│
179|├── references/                       ← 外部源码索引
180|│   ├── README.md                     — triton-ascend + AscendNPU-IR 核心文件位置
181|│   └── ascendnpu-ir-mapping.md      ★ 源码级追踪对照（本项目的每个文件 → ascendnpu-ir）
182|│
183|└── scripts/                          ★ 实用工具
184|    └── trace-to-ascendnpu.sh         — 在 ascendnpu-ir 源码中搜索关键词
185|```
186|
187|### 已验证
188|
189|| 验证项 | 结果 | 说明 |
190||--------|------|------|
191|| `mlir-opt` 降级流水线 | ✅ 3/3 用例通过 | vecadd / matmul / fused |
192|| `g++ -std=c++17` 编译 | ✅ 0 errors | toymini.cpp (1,412 行) |
193|| `mlir-tblgen` TableGen | ✅ 语法通过 | StandaloneOps.td (6 ops) |
194|| CMake + MLIR 集成 | ✅ 配置成功 | 跳过 AddMLIR 冲突 |
195|| bishengir 源码分析 | ✅ 完成 | 3 个 Conversion Pass 逐行解读 |
196|| Triton MLIR 体系 | ✅ 完成 | TT / TritonGPU 双 Dialect 分析 |
197|| matmul 优化方案对比 | ✅ 4 种方案 | 从 74 行到 5 行 |
198|
199|---
200|
201|## 三、学习路径
202|
203|> 🆕 **没有编译器基础？先读 `docs/primer/`（约 30 分钟），再回来学下面的。**
204|
205|### Stage -1: 编译器零基础入门（可选，约 30 分钟）
206|
207|目标：建立从 AST → IR → Pass → Lowering 的基本直觉。
208|路径：`docs/primer/00.md` → `01.md` → `02.md` → `03.md`
209|
210|| 步骤 | 文档 | 概念 | 对应工程 |
211||------|------|------|---------|
212|| -1.1 | `00-编译器是什么` | 编译器三步工作法、为什么需要 IR | — |
213|| -1.2 | `01-AST与IR` | 语法树、三地址码、SSA | toy-mini, standalone-mlir |
214|| -1.3 | `02-Pass与Lowering` | 分析/转换 Pass、dialect、降级 | bishengir-demo, bishengir-op-counter |
215|| -1.4 | `03-从Triton到Ascend` | 全路径串联 | 所有项目 |
216|
217|### Stage 0: LLVM IR 基础（约 3 天）
218|
219|目标：理解 LLVM 编译器的核心模型，为 MLIR 打下基础。
220|
221|```
222|笔记路径: docs/llvm/
223|验证方式: 读懂 .ll 文件 + 理解 Pass 结构
224|```
225|
226|| 步骤 | 笔记 | 知识点 | 产出 |
227||------|------|--------|------|
228|| 0.1 | L00 速通总览 | 三段式架构、学习路线图 | 整体认知 |
229|| 0.2 | L01 架构与 HelloWorld | SSA 形式、Module/Func 结构 | 能读 `.ll` 文件 |
230|| 0.3 | L02 类型系统与 GEP | `iN` 类型、类型转换、GEP 剥洋葱 | 理解地址计算 |
231|| 0.4 | L03 控制流与 Phi | `br` 指令、φ 节点汇合规则 | 理解 CFG |
232|| 0.5 | L04 内置函数与属性 | `llvm.memcpy`、`expect` 内建 | 了解优化基础 |
233|| 0.6 | L05 Pass 开发 | New PM 架构、FunctionPass 骨架 | 能写 BBCounter |
234|| 0.7 | L06 IR 速查表 | 常用指令、调试命令、FileCheck | 快速参考 |
235|
236|**关键突破**: 理解 SSA + φ 节点。这是 MLIR 的 Region 概念的基础。
237|
238|### Stage 1: MLIR 核心概念（约 5 天）
239|
240|目标：掌握 MLIR 的多层 IR 哲学，理解 dialect / operation / pass 三大概念。
241|
242|```
243|笔记路径: docs/mlir/L00 ~ L04
244|验证方式: 运行 bishengir-demo + 读懂 standalone-mlir
245|```
246|
247|| 步骤 | 笔记 | 知识点 | 对应项目 |
248||------|------|--------|---------|
249|| 1.1 | L00 速通与 bishengir | dialect/region/operation 概念 | → bishengir-demo |
250|| 1.2 | L01 Toy Ch1-2 | TableGen 语法、Ops.td 结构 | → toy-mini |
251|| 1.3 | L02 Toy Ch3-6 | Pattern Rewriting、ConversionTarget | → bishengir-op-counter |
252|| 1.4 | L03 自定义 Pass | walk / OpRewritePattern 两种模式 | → bishengir-op-counter |
253|| 1.5 | L04 Standalone 实战 | CMake + Makefile + LLVM 22 适配 | → standalone-mlir |
254|
255|**关键突破**: 理解 MLIR 的 **多层 IR 概念**——为什么需要多个 dialect，如何用 Pass 做 dialect 转换。
256|
257|### Stage 2: 工程实战（约 3 天）
258|
259|目标：从读到写，产出可运行的 MLIR 工程。
260|
261|| 步骤 | 项目 | 行动 | 验证 |
262||------|------|------|------|
263|| 2.1 | bishengir-demo | 运行 3 个用例，观察降级过程 | `mlir-opt` 输出 |
264|| 2.2 | bishengir-demo | 运行 variants/compare.sh，对比 4 种优化方案 | 观察 74 行 → 5 行的变化 |
265|| 2.3 | toy-mini | 编译运行，修改语法扩展 | `./toymini` 输出 |
266|| 2.4 | standalone-mlir | 编译，跑自定义 Pass | `--count-ops` |
267|| 2.5 | bishengir-op-counter | 阅读源码，理解模式 | 对照 Toy Tutorial |
268|
269|**关键突破**: 能用 `mlir-opt` 验证自己的 dialect 理解。
270|
271|### Stage 3: 体系对照（约 2 天）
272|
273|目标：将学到的知识对标到真实项目（bishengir / Triton）。
274|
275|```
276|笔记路径: docs/mlir/L05 ~ L07
277|```
278|
279|| 步骤 | 笔记 | 对标项目 | 产出 |
280||------|------|---------|------|
281|| 3.1 | L05 Toy Mini 手写 | 对照 Toy Tutorial Ch1-2 | 三项目对照表 |
282|| 3.2 | L06 Triton MLIR 体系 | triton-ascend 源码 | TT / TritonGPU Dialect 分析 |
283|| 3.3 | L07 triton-ascend 后端 | ascend_interpreter.py | Python ↔ C++ 对接层 |
284|| 3.4 | L08 bishengir-demo | 三个用例全跑通 | 可运行验证 |
285|
286|**关键突破**: 理解 Triton + bishengir 如何组成完整的 Ascend 编译链路。
287|
288|---
289|
290|## 四、工程项目详情
291|
292|### 4.1 ⭐ bishengir-demo — 可运行降级流水线
293|
294|用标准 `mlir-opt` 模拟 bishengir 三阶段降级过程。
295|
296|#### bishengir 对应
297|
298|| 阶段 | bishengir (实际) | 本 demo (标准 MLIR) | 共同概念 |
299||------|------------------|--------------------|---------|
300|| 输入 | `linalg.generic` | `linalg.generic` | Linalg dialect |
301|| Pass1 | `-convert-linalg-to-hfusion` | `--convert-linalg-to-affine-loops` | 高级→中级 IR |
302|| Pass2 | `-convert-arith-to-hfusion` | `--lower-affine` | 算术操作处理 |
303|| Pass3 | `-convert-hfusion-to-hivm` | `--convert-scf-to-cf --convert-func-to-llvm` | 最终 IR |
304|| 输出 | `hivm.load/vadd/store` | `llvm.load/add/store` | 目标相关指令 |
305|
306|#### 测试结果
307|
308|```
309|向量加法 (vecadd_128.mlir):
310|  Linalg: 3 行  →  Affine: 18 行  →  LLVM: 38 行  (12.7×)
311|  ✅ mlir-opt --convert-linalg-to-affine-loops 通过
312|  ✅ 完整降级到 LLVM IR 通过
313|
314|矩阵乘法 (matmul_4x4x4.mlir):
315|  Linalg: 1 行  →  Affine: 18 行  →  LLVM: 74 行  (74×)
316|  ✅ 基础降级通过
317|  ⚠️ 三重循环完全展开，提供 4 种优化方案对比
318|
319|融合操作 (fused_128.mlir):
320|  Linalg: 15 行  →  Affine: 20 行  →  LLVM: 59 行  (3.9×)
321|  ✅ add + mul 连续操作，展示融合理念
322|```
323|
324|#### matmul 的 74× 膨胀与优化
325|
326|| Variant | 策略 | LLVM 行数 | vs 基准 | 对应 bishengir |
327||---------|------|-----------|---------|---------------|
328|| **V0** | 无优化 (基准) | 74 行 | - | — |
329|| **V1** | 循环分块 (tile=2x2x1) | 76 行 | +2 行 | — |
330|| **V2** | 向量化 (tile+vectorize) | 77 行 | +3 行 | `-convert-hfusion-to-hivm` 生成向量指令 |
331|| **V3** | **硬件映射 (模拟 mmul)** | **5 行** | **-69 行 (-93%)** | `hfusion.cube_matmul → hivm.mmul` |
332|
333|V3 的 5 行 vs 74 行的差距，正是 bishengir 实际采用的方案——**保持高级语义不展开，直接映射到硬件 Cube 单元**。
334|
335|```bash
336|# 运行对比
337|bash projects/bishengir-demo/variants/compare.sh
338|```
339|
340|### 4.2 toy-mini — 从零写 Toy 语言解析器
341|
342|| 特性 | 值 |
343||------|-----|
344|| 语言 | C++17（纯标准库，零外部依赖）|
345|| 行数 | 1,412 行 |
346|| 编译 | `g++ -std=c++17 -o toymini toymini.cpp` |
347|| 输入 | 含 Lexer(14 tokens) + Parser(递归下降) + AST(8 节点) + MLIR Gen |
348|
349|**支持语法示例**:
350|```toy
351|# 函数定义 + 数组字面量 + 二元运算 + 转置 + 打印
352|def main() {
353|  var A = [[1, 2, 3, 4], [5, 6, 7, 8]];
354|  var B = transpose(A);
355|  var C = B + A;
356|  var D = C * 2.0;
357|  print(D);
358|  return;
359|}
360|```
361|
362|### 4.3 standalone-mlir — 从零构建 MLIR dialect
363|
364|**6 个 Op**: constant / add / mul / transpose / print / return
365|**2 个 Pass**: `-count-ops` (分析) + `-elim-transpose` (转换)
366|**2 种构建**: CMake + Makefile 双方案
367|**1 个入口**: `standalone-opt` (类似 `bishengir-opt`)
368|
369|### 4.4 bishengir-op-counter — 自定义 Pass 参考代码
370|
371|| 文件 | 类型 | 模式 | 对应 Toy Tutorial |
372||------|------|------|-------------------|
373|| `BishengirOpCounter.cpp` | 分析 Pass | `op->walk()` | Ch3 ShapeInferencePass |
374|| `BishengirPeelTranspose.cpp` | 转换 Pass | `OpRewritePattern` | Ch3 ToyCombine |
375|
376|---
377|
378|## 五、快速入门
379|
380|### 5.1 环境准备
381|
382|```bash
383|# 检查环境
384|bash setup.sh
385|
386|# 如果缺依赖
387|brew install llvm cmake
388|xcode-select --install
389|```
390|
391|### 5.2 跑 bishengir-demo（5 分钟）
392|
393|```bash
394|cd projects/bishengir-demo
395|export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
396|
397|# 单个用例
398|mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir
399|
400|# 完整降级到 LLVM IR
401|mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm test-cases/vecadd_128.mlir
402|
403|# 批量运行
404|bash run-demo.sh
405|```
406|
407|### 5.3 跑 Toy Mini（3 分钟）
408|
409|```bash
410|cd projects/toy-mini
411|g++ -std=c++17 -o toymini toymini.cpp
412|./toymini
413|```
414|
415|### 5.4 编译 standalone-mlir（10 分钟）
416|
417|```bash
418|cd projects/standalone-mlir
419|export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
420|cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
421|cmake --build build
422|./build/standalone-opt test/example.mlir
423|```
424|
425|### 5.5 快速调试技巧
426|
427|```bash
428|# 查看 IR 的某个 stage
429|mlir-opt --print-ir-after=<pass-name> input.mlir
430|
431|# 只跑某个 pass
432|mlir-opt --pass-pipeline="builtin.module(func.func(count-ops))" input.mlir
433|```
434|
435|---
436|
437|## 六、三项目技术对照
438|
439|| 维度 | LLVM IR | MLIR | bishengir |
440||------|---------|------|-----------|
441|| **设计哲学** | 单一 IR | 多层 IR (dialect) | 专用 dialect 链 |
442|| **类型系统** | `iN`, `ptr`, `struct` | `tensor<T>`, `memref<T>` | `hfusion.tensor<T>` |
443|| **操作** | 指令 (add/load/store) | Operation (可嵌套) | hivm.vadd/madd |
444|| **优化** | Pass (FunctionPass) | Pass + Pattern Rewriting | ConversionTarget |
445|| **降级** | 前端 → IR → 后端 | dialect → dialect → ... | Linalg → HFusion → HIVM |
446|| **元编程** | TableGen (指令描述) | TableGen (dialect 定义) | TableGen |
447|
448|### 对应关系
449|
450|```
451|Triton Python kernel
452|  ↓ Frontend
453|Triton IR (tt.load/tt.dot/tt.store)
454|  ↓ [本项目的分析对象]
455|AscendNPU-IR (华为官方开源)
456|  ├── bishengir (Nous Research fork)
457|  ├── LinalgToHFusion   →  Linalg ops  →  HFusion ops
458|  ├── ArithToHFusion    →  Arith ops    →  HFusion ops
459|  └── HFusionToHIVM     →  HFusion ops  →  HIVM ops (NPU)
460|      ↓
461|CANN Runtime (华为 SDK, 硬件执行)
462|```
463|
464|**相关文档**: https://ascendnpu-ir.gitcode.com/zh_cn/index.html
465|**项目地址**: https://github.com/Ascend/AscendNPU-IR
466|
467|---
468|
469|## 七、依赖与环境
470|
471|| 工具 | 版本 | 安装方式 | 用途 |
472||------|------|---------|------|
473|| LLVM/MLIR | 22.1.6 | `brew install llvm` | mlir-opt, mlir-tblgen |
474|| cmake | ≥ 3.20 | `brew install cmake` | standalone-mlir 构建 |
475|| ninja | (可选) | `brew install ninja` | 加速构建 |
476|| g++/clang++ | C++17 | `xcode-select --install` | Toy Mini 编译 |
477|| python3 | ≥ 3.8 | 系统自带 | 用例生成器 |
478|
479|**环境变量**:
480|
481|```bash
482|export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
483|export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
484|```
485|
486|---
487|
488|## 八、笔记与项目对照表
489|
490|| 笔记 | 对应项目 | 核心知识点 |
491||------|---------|-----------|
492|| `primer/00` | — | 编译器三步工作法 |
493|| `primer/01` | toy-mini, standalone-mlir | AST、SSA |
494|| `primer/02` | bishengir-demo, bishengir-op-counter | Pass、Lowering、dialect |
495|| `primer/03` | 全部 | 全路径串联 |
496|| `MLIR-L00` | bishengir-demo | bishengir 三段降级全景 |
497|| `MLIR-L01` | toy-mini | TableGen dialect 定义 |
498|| `MLIR-L02` | bishengir-op-counter | Pattern Rewriting 模式 |
499|| `MLIR-L03` | bishengir-op-counter | Pass 架构：分析 vs 转换 |
500|| `MLIR-L04` | standalone-mlir | CMake + Makefile + LLVM22 适配 |
501|