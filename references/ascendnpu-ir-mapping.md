1|# AscendNPU-IR 源码对接追踪
2|
3|> 本文件将本项目的概念/代码/文档与 AscendNPU-IR 官方源码和文档建立对应关系。
4|> 读者拿到 AscendNPU-IR 源码后，可根据此文件快速定位。
5|
6|**参考链接**:
7|- 官方代码仓: https://github.com/Ascend/AscendNPU-IR
8|- 中文文档: https://ascendnpu-ir.gitcode.com/zh_cn/index.html
9|- 本项目分析的 fork: `~/hermes-workspace/ascendnpu-ir/` (bishengir)
10|
11|---
12|
13|## 1. 三阶段降级源码追踪
14|
15|bishengir 的核心是三阶段降级：Linalg → HFusion → HIVM。
16|每个阶段在 ascendnpu-ir 源码中对应一个 Conversion Pass 目录。
17|
18|### 1.1 Pass1: Linalg → HFusion
19|
20|| 项 | 内容 |
21||----|------|
22|| **目录** | `bishengir/lib/Conversion/LinalgToHFusion/` |
23|| **主文件** | `LinalgToHFusion.cpp` |
24|| **核心类** | `ConvertLinalgToHFusion` (Pass 入口) |
25|| **Pattern 模式** | `OpRewritePattern<linalg::GenericOp>` 匹配 linalg 操作 |
26|| **输出** | `hfusion.elemwise_binary` / `hfusion.cube_matmul` |
27|| **匹配条件** | `linalg::GenericOp` 的 body 中只含 `arith.addf` 等基本运算 |
28|| **测试用例** | `test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir` |
29|
30|**关键函数**:
31|
32|```cpp
33|// LinalgToHFusion.cpp 中的核心 Pattern
34|class LinalgGenericToHFusion : public OpRewritePattern<linalg::GenericOp> {
35|    LogicalResult matchAndRewrite(linalg::GenericOp op, ...) override {
36|        // 1. 验证: linalg.generic 是否满足转换条件
37|        // 2. 根据 body 中的运算选择 hfusion op 类型
38|        //    arith.addf → hfusion.elemwise_binary {fun = add}
39|        //    arith.mulf → hfusion.elemwise_binary {fun = mul}
40|        // 3. 生成 hfusion op，替换原 linalg.generic
41|    }
42|};
43|```
44|
45|### 1.2 Pass2: Arith → HFusion
46|
47|| 项 | 内容 |
48||----|------|
49|| **目录** | `bishengir/lib/Conversion/ArithToHFusion/` |
50|| **主文件** | `ArithToHFusion.cpp` |
51|| **核心类** | `ConvertArithToHFusion` |
52|| **处理 op** | `arith.addf`, `arith.mulf`, `arith.cmpf` 等 |
53|| **测试用例** | `test/Conversion/ArithToHFusion/arith-to-hfusion.mlir` |
54|
55|### 1.3 Pass3: HFusion → HIVM
56|
57|| 项 | 内容 |
58||----|------|
59|| **目录** | `bishengir/lib/Conversion/HFusionToHIVM/` |
60|| **主文件** | `HFusionToHIVM.cpp` |
61|| **核心类** | `ConvertHFusionToHIVM` |
62|| **输出** | `hivm.load` / `hivm.vadd` / `hivm.mmul` / `hivm.store` |
63|| **测试用例** | `test/Conversion/HFusionToHIVM/hfusion-to-hivm.mlir` |
64|
65|**关键区别**:
66|- HFusion 是**算子级 IR**（`hfusion.elemwise_binary` 表示"执行逐元素运算"）
67|- HIVM 是**指令级 IR**（`hivm.vadd` 表示"执行向量加法指令"）
68|- 类比: HFusion = "做一盘炒鸡蛋"，HIVM = "打蛋→热油→炒→装盘"
69|
70|---
71|
72|## 2. Dialect 源码追踪
73|
74|### 2.1 HFusion Dialect
75|
76|| 项 | 内容 |
77||----|------|
78|| **定义文件** | `bishengir/include/bishengir/Dialect/HFusion/HFusionOps.td` |
79|| **实现文件** | `bishengir/lib/Dialect/HFusion/` |
80|| **核心 Op** | `elemwise_binary`, `elemwise_unary`, `cube_matmul` |
81|| **TableGen 基类** | `HFusion_Op<string mnemonic>` (类似 Toy Tutorial 的 `Toy_Op`) |
82|
83|```
84|        HFusionOps.td
85|        ├── def ElemwiseBinaryOp
86|        │   ├── 参数: lhs, rhs, fun (add/mul/sub/div)
87|        │   ├── 输出: result tensor
88|        │   └── 约束: lhs 和 rhs 形状相同
89|        ├── def ElemwiseUnaryOp
90|        │   ├── 参数: input, fun (exp/sqrt/neg)
91|        │   └── 输出: result tensor
92|        └── def CubeMatmulOp
93|            ├── 参数: A, B, C (memref)
94|            └── 语义: C += A × B (矩阵乘加)
95|```
96|
97|### 2.2 HIVM Dialect
98|
99|| 项 | 内容 |
100||----|------|
101|| **定义文件** | `bishengir/include/bishengir/Dialect/HIVM/HIVMOps.td` |
102|| **实现文件** | `bishengir/lib/Dialect/HIVM/` |
103|| **核心 Op** | `vload`, `vadd`, `vmul`, `vstore`, `mmul` |
104|| **语义** | 每条指令对应一个 Ascend NPU 硬件指令 |
105|
106|### 2.3 Annotation / HACC / Scope / Symbol Dialect
107|
108|| Dialect | 定义文件 | 用途 |
109||---------|---------|------|
110|| **Annotation** | `Dialect/Annotation/AnnotationOps.td` | 标记/注释（用于调试、profiling） |
111|| **HACC** | `Dialect/HACC/HACCOps.td` | 高级计算控制（循环/同步/调度） |
112|| **Scope** | `Dialect/Scope/ScopeOps.td` | 作用域管理（内存区域/执行域） |
113|| **Symbol** | `Dialect/Symbol/SymbolOps.td` | 符号管理（变量名/函数名映射） |
114|
115|---
116|
117|## 3. 翻译文档索引
118|
119|以下 9 篇中文翻译文档在 `docs/ascendnpu-ir/translations/` 目录下。
120|
121|| # | 文档 | 原始源码路径 | 内容 |
122||---|------|------------|------|
123|| 01 | `AnnotationPasses.md` | `bishengir/docs/cn/Pass/AnnotationPass.md` | Annotation dialect 的转换 Pass |
124|| 02 | `HACCPasses.md` | `bishengir/docs/cn/Pass/HACCPass.md` | HACC dialect 的转换 Pass |
125|| 03 | `ScopePasses.md` | `bishengir/docs/cn/Pass/ScopePass.md` | Scope dialect 的转换 Pass |
126|| 04 | `SymbolPasses.md` | `bishengir/docs/cn/Pass/SymbolPass.md` | Symbol dialect 的转换 Pass |
127|| 05 | `AnnotationDialect.md` | `bishengir/docs/cn/Dialect/AnnotationDialect.md` | Annotation dialect 定义详解 |
128|| 06 | `ScopeDialect.md` | `bishengir/docs/cn/Dialect/ScopeDialect.md` | Scope dialect 定义详解 |
129|| 07 | `SymbolDialect.md` | `bishengir/docs/cn/Dialect/SymbolDialect.md` | Symbol dialect 定义详解 |
130|| 08 | `MathExtDialect.md` | `bishengir/docs/cn/Dialect/MathExtDialect.md` | 数学扩展 dialect |
131|| 09 | `MemRefExtDialect.md` | `bishengir/docs/cn/Dialect/MemRefExtDialect.md` | 内存引用扩展 dialect |
132|
133|---
134|
135|## 4. 深度分析笔记索引
136|
137|以下 2 篇深度分析笔记在 `docs/ascendnpu-ir/analysis/` 目录下。
138|
139|| 文档 | 大小 | 内容概要 |
140||------|------|---------|
141|| `BishengIR代码仓库解读.md` | 21.5KB | 代码仓库结构逐目录解读，含每个子目录的功能说明和关键文件清单 |
142|| `AscendNPUIR文档总结.md` | 32KB | 官方文档体系的完整总结，含架构图、Pass 管线总览、dialect 关系图 |
143|
144|---
145|
146|## 5. 阅读路径
147|
148|拿到 AscendNPU-IR 源码后，按以下顺序阅读：
149|
150|```
151|Step 1: 入口 ── tools/bishengir-opt/bishengir-opt.cpp
152|  看 Pass 怎么注册、dialect 怎么加载、命令行怎么解析
153|
154|Step 2: 一个完整 Pass ── lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
155|  看 ConversionTarget 怎么设、RewritePattern 怎么写、applyPartialConversion 怎么调
156|
157|Step 3: Dialect 定义 ── include/bishengir/Dialect/HFusion/HFusionOps.td
158|  看 TableGen 怎么定义 Op、assemblyFormat 怎么写、constraints 怎么加
159|
160|Step 4: 测试用例 ── test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
161|  看输入 IR 和输出 IR 的格式，理解 Pass 的效果
162|
163|Step 5: 自定义 Pass ── 对照本项目的 bishengir-op-counter
164|  按 bishengir-op-counter 的注释提示注册到 bishengir-opt
165|```
166|
167|**对应本项目的学习路径**:
168|
169|| 本项目 Stage | 对应 AscendNPU-IR 阅读 |
170||-------------|----------------------|
171|| Stage -1 (Primer) | — |
172|| Stage 0 (LLVM IR) | — |
173|| Stage 1 (MLIR 概念) | 读本 mapping 文档 §1 |
174|| Stage 2 (工程实战) | 读 Step 1→4，跑测试用例 |
175|| Stage 3 (体系对照) | 读 §2 Dialect 源码 + §3 翻译文档 |
176|
177|---
178|
179|## 6. 项目工程源码追踪
180|
181|### 6.1 bishengir-demo ↔ AscendNPU-IR
182|
183|| 本项目的文件 | 对应 AscendNPU-IR 源码 | 关系 |
184||------------|----------------------|------|
185|| `test-cases/vecadd_128.mlir` | `test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir` | 同类型输入，本 demo 用标准 MLIR |
186|| `test-cases/matmul_4x4x4.mlir` | `test/Conversion/LinalgToHFusion/matmul-to-hfusion.mlir` | 同类型输入 |
187|| `test-cases/fused_128.mlir` | 无直接对应 | 演示融合概念 |
188|| `variants/variant0_baseline.sh` | `bishengir-opt --convert-linalg-to-hfusion` | 等价命令行 |
189|| `variants/variant3_hw_mapping.sh` | `lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp` | 模式对照 |
190|
191|### 6.2 bishengir-op-counter ↔ AscendNPU-IR
192|
193|| 本项目的文件 | 对应 AscendNPU-IR 源码 | 模式关系 |
194||------------|----------------------|---------|
195|| `BishengirOpCounter.cpp` | `lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp` | 分析 Pass vs 转换 Pass |
196|| `BishengirPeelTranspose.cpp` | `lib/Conversion/ToyCombine.cpp` | 同使用 `OpRewritePattern` |
197|
198|### 6.3 standalone-mlir ↔ AscendNPU-IR
199|
200|| 本项目的文件 | 对应 AscendNPU-IR 源码 | 模式关系 |
201||------------|----------------------|---------|
202|| `include/standalone/StandaloneOps.td` | `include/bishengir/Dialect/HFusion/HFusionOps.td` | 同 TableGen 语法 |
203|| `tools/standalone-opt.cpp` | `tools/bishengir-opt/bishengir-opt.cpp` | 同入口模式 |
204|| `CMakeLists.txt` | `bishengir/CMakeLists.txt` | 同 `find_package(MLIR)` 模式（本 demo 跳过）|
205|
206|---
207|
208|## 7. AscendNPU-IR 源码目录速查
209|
210|| 目录 | 功能 | 在本项目中的对应 |
211||------|------|----------------|
212|| `tools/bishengir-opt/` | 主入口，类似 mlir-opt | `standalone-mlir/tools/standalone-opt.cpp` |
213|| `include/bishengir/Dialect/` | Dialect 的 .td 定义 | `standalone-mlir/include/standalone/StandaloneOps.td` |
214|| `lib/Dialect/HFusion/` | HFusion dialect 实现 | — |
215|| `lib/Dialect/HIVM/` | HIVM dialect 实现 | — |
216|| `lib/Conversion/LinalgToHFusion/` | Pass1 实现 | bishengir-demo Stage1 |
217|| `lib/Conversion/ArithToHFusion/` | Pass2 实现 | bishengir-demo Stage2 |
218|| `lib/Conversion/HFusionToHIVM/` | Pass3 实现 | bishengir-demo Stage3 |
219|| `test/Conversion/` | 测试用例 | bishengir-demo test-cases |
220|| `docs/cn/` | 中文文档 | `docs/ascendnpu-ir/translations/` |
221|