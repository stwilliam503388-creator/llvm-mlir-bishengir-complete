1|1|# bishengir-demo — 可运行 MLIR 降级流水线
2|2|
3|3|用标准 `mlir-opt` 模拟 bishengir 三阶段降级（Linalg → HFusion → HIVM）。
4|4|
5|5|---
6|6|
7|7|## 测试用例（28 个）— 按难度分级
8|8|
9|9|> ⭐ = 入门（读完 Primer 就能看）
10|10|> ⭐⭐ = 进阶（需要 LLVM IR 基础）
11|11|> ⭐⭐⭐ = 复杂（需要 MLIR dialect / lowering 概念）
12|12|
13|13|> 💡 **遇到不认识的术语？** → 查 `docs/reference/技术术语速查手册.md`（298 条，每条含"一句话"+"类比"）
14|
15|### 目录结构
16|14|
17|15|```
18|16|test-cases/
19|17|├── mlir/01_basic/    ← ⭐ 入门 8 个
20|18|├── mlir/02_intermediate/ ← ⭐⭐ 进阶 11 个
21|19|└── mlir/03_advanced/     ← ⭐⭐⭐ 复杂 9 个
22|20|```
23|21|
24|22|| 难度 | 数量 | 涉及概念 | 适合谁 |
25|23||------|------|---------|--------|
26|24|| ⭐ 入门 | 8 | `linalg.generic` + parallel | 读完 Primer 即可 |
27|25|| ⭐⭐ 进阶 | 11 | `reduction` iterator、组合模式、条件分支 | 有 LLVM IR 基础 |
28|26|| ⭐⭐⭐ 复杂 | 9 | `linalg.matmul`、named op、`affine.for`、多步 pipeline | 理解 MLIR dialect 后 |
29|27|
30|28|> ⭐ = 入门（读完 Primer 就能看）
31|29|> ⭐⭐ = 进阶（需要 LLVM IR 基础）
32|30|> ⭐⭐⭐ = 复杂（需要 MLIR dialect / lowering 概念）
33|31|
34|32|---
35|33|
36|34|### 第一梯队：入门 ⭐（8 个）
37|35|
38|36|最基础的逐元素运算，只需理解 `linalg.generic` 的 "parallel" iterator 即可。
39|37|
40|38|#### vecadd_128（向量加法）— 残差连接 / shortcut
41|39|
42|40|| 维度 | 内容 |
43|41||------|------|
44|42|| **功能** | `C[i] = A[i] + B[i]`，逐元素向量加法 |
45|43|| **AI 角色** | **残差连接 (Residual Connection/Shortcut)**：ResNet、Transformer 每层输出 + 输入直接相加，解决深层网络梯度消失。LLM 中每个 Attention/FFN 层后都有 `x + sublayer(x)` |
46|44|| **MLIR 模式** | `linalg.generic` + `arith.addf`，3 行 → 38 行 LLVM（12.7×）|
47|45|| **对应 bishengir** | `hfusion.elemwise_binary {fun = add}` |
48|46|
49|47|#### relu_4x4（ReLU 激活）— 全模型通用激活函数
50|48|
51|49|| 维度 | 内容 |
52|50||------|------|
53|51|| **功能** | `y = max(0, x)`，负值截断为 0 |
54|52|| **AI 角色** | **ReLU (Rectified Linear Unit)**：CNN 全系标配，计算量最低的激活函数，GPU 友好。LLM 中 FFN 层用 ReLU 变体 (GELU/SwiGLU) 替代 |
55|53|| **MLIR 模式** | `arith.cmpf` + `arith.select`，条件分支 |
56|54|| **对应 bishengir** | `hfusion.elemwise_unary {fun = relu}` |
57|55|
58|56|#### tanh_4（Tanh 激活）— RNN / LSTM
59|57|
60|58|| 维度 | 内容 |
61|59||------|------|
62|60|| **功能** | `y = tanh(x)`，S 形函数，输出范围 (-1, 1) |
63|61|| **AI 角色** | **Tanh (Hyperbolic Tangent)**：RNN/LSTM 的默认激活函数，用于控制信息流。LLM 中较少使用，但在某些门控机制中仍有出现 |
64|62|| **MLIR 模式** | `math.tanh` 内建函数调用 |
65|63|| **对应 bishengir** | `hfusion.elemwise_unary {fun = tanh}` |
66|64|
67|65|#### softmax_4（Softmax 指数）— Attention 核心
68|66|
69|67|| 维度 | 内容 |
70|68||------|------|
71|69|| **功能** | `y = exp(x)`，指数运算（softmax 的前半部分）|
72|70|| **AI 角色** | **Softmax 指数**：Transformer 的 Attention 机制核心——计算 query 和 key 的匹配分数。完整 softmax = `exp(x - max(x)) / Σexp(x - max(x))`，本文件只演示 exp 部分 |
73|71|| **MLIR 模式** | `math.exp` 内建函数 |
74|72|| **对应 bishengir** | 逐元素 `math` 操作映射 |
75|73|
76|74|#### broadcast_4x4（标量广播）— Bias 加法
77|75|
78|76|| 维度 | 内容 |
79|77||------|------|
80|78|| **功能** | `B[i][j] = A`，把标量复制到矩阵每个位置 |
81|79|| **AI 角色** | **Broadcasting (广播)**：神经网络的基础操作——卷积层的 bias、Batch Norm 的 γ/β、Attention 中的位置编码都需要 broadcast 后与特征图相加 |
82|80|| **MLIR 模式** | `affine_map<(i,j) -> ()>`，标量→矩阵 |
83|81|| **对应 bishengir** | `hfusion.broadcast` |
84|82|
85|83|#### dropout_4x4（Dropout 训练）— 防止过拟合
86|84|
87|85|| 维度 | 内容 |
88|86||------|------|
89|87|| **功能** | `y = x * scale`，训练时按概率缩放（简化版，不含 mask）|
90|88|| **AI 角色** | **Dropout (随机丢弃)**：训练时随机忽略部分神经元，防止过拟合。Transformer 早期使用（BERT），现代 LLM (GPT-4/LLaMA) 趋向于不用 |
91|89|| **MLIR 模式** | `arith.mulf` 逐元素乘法 |
92|90|| **对应 bishengir** | `hfusion.elemwise_binary {fun = mul}` |
93|91|
94|92|#### fill_4x4（张量填充）— 初始化缓冲区
95|93|
96|94|| 维度 | 内容 |
97|95||------|------|
98|96|| **功能** | 用常数值填充整个张量 |
99|97|| **AI 角色** | **初始化 (Initialization)**：在卷积/矩阵乘前初始化输出缓冲区。几乎每个模型的第一层和中间层都会用到 |
100|98|| **MLIR 模式** | `linalg.generic` + `yield %cst` |
101|99|| **对应 bishengir** | `linalg.fill`（Homebrew 未编译，用 generic 替代）|
102|100|
103|101|#### fused_128（add + mul 融合）— 算子融合概念
104|102|
105|103|| 维度 | 内容 |
106|104||------|------|
107|105|| **功能** | 连续两个 `linalg.generic`：先加后乘 |
108|106|| **AI 角色** | **算子融合 (Kernel Fusion)**：编译器核心优化——将连续两个 kernel 合并为一个，减少内存读写。深度学习编译器 (TVM/XLA) 的核心能力 |
109|107|| **MLIR 模式** | 连续 `linalg.generic` 两次 |
110|108|| **对应 bishengir** | HFusion 的"算子融合"概念演示 |
111|109|
112|110|---
113|111|
114|112|### 第二梯队：进阶 ⭐⭐（11 个）
115|113|
116|114|需要理解 `reduction` iterator、`arith.cmpf` 条件分支、以及连续多步运算。
117|115|
118|116|#### sigmoid_4（Sigmoid 激活）— 二分类 / RNN 门控
119|117|
120|118|| 维度 | 内容 |
121|119||------|------|
122|120|| **功能** | `σ(x) = 1 / (1 + e^{-x})`，输出范围 (0, 1) |
123|121|| **AI 角色** | **Sigmoid (Logistic 函数)**：二分类输出层、RNN 的 forget/input/output 门控。LLaMA 等现代 LLM 也用它做门控激活 (SwiGLU) |
124|122|| **MLIR 模式** | `arith.negf` + `math.exp` + `arith.addf` + `arith.divf`，4 步组合 |
125|123|| **难度提示** | 需要理解 `arith.negf` 和 `math.exp` 的配合 |
126|124|
127|125|#### silu_4（SiLU / Swish 激活）— LLaMA 系列
128|126|
129|127|| 维度 | 内容 |
130|128||------|------|
131|129|| **功能** | `silu(x) = x * σ(x)`，Sigmoid 门控的输入 |
132|130|| **AI 角色** | **SiLU (Sigmoid Linear Unit)**：LLaMA 2/3、Mistral、Gemma 等现代 LLM 的 FFN 层使用 SwiGLU (SiLU 的变体)。比 ReLU 平滑，比 GELU 计算量低 |
133|131|| **MLIR 模式** | `sigmoid` + `arith.mulf`，5 步组合 |
134|132|| **对应 bishengir** | 组合模式→可融合为单个 hfusion |
135|133|
136|134|#### leaky_relu_4（Leaky ReLU）— GAN
137|135|
138|136|| 维度 | 内容 |
139|137||------|------|
140|138|| **功能** | `y = x if x > 0 else 0.01*x`，负数侧有微小斜率 |
141|139|| **AI 角色** | **Leaky ReLU (带泄漏的线性整流)**：解决 ReLU 死亡问题（负数区梯度为零）。GAN (生成对抗网络) 标配，部分传统 CNN 也使用 |
142|140|| **MLIR 模式** | `arith.cmpf` + `arith.mulf` + `arith.select` |
143|141|| **对应 bishengir** | 条件分支模式 |
144|142|
145|143|#### gelu_tanh_4（GELU 近似）— BERT / GPT
146|144|
147|145|| 维度 | 内容 |
148|146||------|------|
149|147|| **功能** | `gelu(x) ≈ 0.5 * x * (1 + tanh(x))` |
150|148|| **AI 角色** | **GELU (Gaussian Error Linear Unit)**：BERT/GPT-2/GPT-3 的 FFN 激活函数。比 ReLU 平滑，性能更好。GPT-4 和 LLaMA 改用 SwiGLU |
151|149|| **MLIR 模式** | `math.tanh` + `arith.addf` + `arith.mulf`，4 步组合 |
152|150|| **对应 bishengir** | 组合模式可融合 |
153|151|
154|152|#### hard_sigmoid_4（Hard Sigmoid）— MobileNet
155|153|
156|154|| 维度 | 内容 |
157|155||------|------|
158|156|| **功能** | `hard_sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)`，sigmoid 的线性近似 |
159|157|| **AI 角色** | **Hard Sigmoid (硬 Sigmoid)**：MobileNetV3 等轻量化模型使用，计算量比 sigmoid 小 3×。适合移动端部署 |
160|158|| **MLIR 模式** | `arith.maximumf` + `arith.minimumf`，数值裁剪 |
161|159|| **对应 bishengir** | 分段线性函数映射 |
162|160|
163|161|#### prelu_4x4（PReLU）— 图像超分辨率
164|162|
165|163|| 维度 | 内容 |
166|164||------|------|
167|165|| **功能** | `y = x if x > 0 else α*x`，α 是可训练参数 |
168|166|| **AI 角色** | **PReLU (Parametric ReLU, 参数化线性整流)**：Leaky ReLU 的扩展，α 通过梯度下降学习。ESPCN/SRGAN 等超分辨率模型常用 |
169|167|| **MLIR 模式** | `arith.mulf` + `arith.cmpf` + `arith.select` |
170|168|
171|169|#### reduce_sum_4x4（求和归约）— Layer Norm 分母
172|170|
173|171|| 维度 | 内容 |
174|172||------|------|
175|173|| **功能** | `sum = ΣᵢΣⱼ x[i][j]`，矩阵所有元素求和 |
176|174|| **AI 角色** | **Sum Reduction (求和归约)**：Layer Norm 需要计算 `mean = Σx / N` 和 `variance = Σ(x - mean)² / N`，求和是第一步 |
177|175|| **MLIR 模式** | **`reduction` iterator**，多维→标量 |
178|176|| **对应 bishengir** | `hfusion.reduce {fun = add, axes = [0, 1]}` |
179|177|| **难度提示** | 这是第一个 `reduction` 用法的例子，理解后其他 reduction 都类似 |
180|178|
181|179|#### reduce_max_4x4（最大值归约）— Softmax 数值稳定
182|180|
183|181|| 维度 | 内容 |
184|182||------|------|
185|183|| **功能** | `max = max(all elements)`，求矩阵最大值 |
186|184|| **AI 角色** | **Max Reduction (最大值归约)**：softmax 数值稳定性关键——`softmax(x) = exp(x - max(x)) / Σexp(x - max(x))`，先减去最大值防止 exp 溢出 |
187|185|| **MLIR 模式** | `reduction` + `arith.cmpf` + `arith.select` |
188|186|
189|187|#### softmax_complete_4（完整 Softmax 第一步）— Attention 核心
190|188|
191|189|| 维度 | 内容 |
192|190||------|------|
193|191|| **功能** | `y = exp(x - max(x))`，减最大值后取指数 |
194|192|| **AI 角色** | **数值稳定 Softmax**：Attention 机制的完整第一步。Transformer 中的 `softmax(QK^T / √d)` 依赖于数值稳定的 exp |
195|193|| **MLIR 模式** | `memref.alloc` + `reduction` + `memref.store` |
196|194|| **难度提示** | 需要理解 memref 的手动分配和写入 |
197|195|
198|196|#### clamp_4x4（数值裁剪）— 梯度裁剪
199|197|
200|198|| 维度 | 内容 |
201|199||------|------|
202|200|| **功能** | `y = clamp(x, min, max)`，限制值在 [min, max] 区间 |
203|201|| **AI 角色** | **Gradient Clipping (梯度裁剪)**：训练时限制梯度范围，防止梯度爆炸。LLM 训练必用。推理时可用于激活值截断（量化友好） |
204|202|| **MLIR 模式** | 两次 `arith.cmpf` + `arith.select` |
205|203|| **对应 bishengir** | 分段线性函数映射 |
206|204|
207|205|---
208|206|
209|207|### 第三梯队：复杂 ⭐⭐⭐（9 个）
210|208|
211|209|需要理解 `linalg.matmul` named op、多步 pipeline、手动循环 (affine.for)、以及完整的 BN/LayerNorm 计算链。
212|210|
213|211|#### matmul_4x4x4（矩阵乘法）— Linear / MLP 层
214|212|
215|213|| 维度 | 内容 |
216|214||------|------|
217|215|| **功能** | `C = A @ B` (矩阵乘法)，4×4 × 4×4 → 4×4 |
218|216|| **AI 角色** | **Linear Layer (全连接层 / 线性层)**：`y = x @ W^T + b`。LLM 中 Attention 的 Q/K/V 投影、FFN 的 up/down projection 全部是 matmul。**这是 AI 模型最核心的算子，占算力 60-80%** |
219|217|| **MLIR 模式** | `linalg.matmul` named op，1 行 → **74 行 LLVM**（74× 膨胀）|
220|218|| **对应 bishengir** | `hfusion.cube_matmul` → `hivm.mmul`（1 行，硬件指令）|
221|219|| **难度提示** | 74× 膨胀不是问题——bishengir 保持 1 行 NPU 指令。膨胀展示了"不保留语义会怎样" |
222|220|
223|221|#### gemm_relu_4x4（矩阵乘 + ReLU 融合）— MLP 标准模式
224|222|
225|223|| 维度 | 内容 |
226|224||------|------|
227|225|| **功能** | `y = ReLU(x @ W)`，先矩阵乘后激活 |
228|226|| **AI 角色** | **算子融合 (GEMM + Activation)**：MLP 层的标准模式——`x @ W1 + b → ReLU → x @ W2 + b`。编译器可以将 matmul + relu 融合为单个 kernel，减少一次中間 buffer 读写 |
229|227|| **MLIR 模式** | `linalg.matmul` + `linalg.generic` 两阶段 pipeline |
230|228|| **对应 bishengir** | 融合优化 |
231|229|| **难度提示** | 需要理解两阶段 lowering 的配合 |
232|230|
233|231|#### depthwise_conv_4x4（深度卷积）— MobileNet
234|232|
235|233|| 维度 | 内容 |
236|234||------|------|
237|235|| **功能** | 逐通道 3×3 卷积，输入 1×4×4×1，输出 1×4×4×1 |
238|236|| **AI 角色** | **Depthwise Convolution (深度可分离卷积)**：MobileNet/EfficientNet 的核心算子，计算量是标准卷积的 1/C (C 为通道数)。结合 pointwise conv 组成 depthwise separable conv |
239|237|| **MLIR 模式** | `linalg.depthwise_conv_2d_nhwc_hwcm` named op，3 行 → 113 行 LLVM |
240|238|| **对应 bishengir** | named op 映射 |
241|239|| **难度提示** | 113 行 LLVM = 稠密的卷积展开，最复杂的单独 op |
242|240|
243|241|#### conv2d_4x4（二维卷积）— 标准卷积层
244|242|
245|243|| 维度 | 内容 |
246|244||------|------|
247|245|| **功能** | 2D valid 卷积：输入 4×4，kernel 3×3，输出 2×2 |
248|246|| **AI 角色** | **Convolution (卷积)**：CNN 的绝对核心——ResNet/VGG/YOLO/UNet 等视觉模型全部依赖卷积。LLM 中虽然不直接用，但多模态模型 (GPT-4V) 的视觉编码器仍用卷积 |
249|247|| **MLIR 模式** | `linalg.generic` + `reduction` × 2，6 行 → 85 行 LLVM |
250|248|| **对应 bishengir** | 可被 bishengir 模式匹配优化 |
251|249|| **难度提示** | 需要理解 `reduction` iterator 与 affine_map 的配合 |
252|250|
253|251|#### max_pool_4x4（最大池化）— 下采样
254|252|
255|253|| 维度 | 内容 |
256|254||------|------|
257|255|| **功能** | 2×2 窗口内取最大值，stride=2，4×4 → 2×2 |
258|256|| **AI 角色** | **Max Pooling (最大池化)**：CNN 的下采样层——保留最强激活值，丢弃位置信息。经典 CNN (LeNet/AlexNet/VGG) 标配，现代模型趋向于用 stride=2 的卷积替代 |
259|257|| **MLIR 模式** | `affine.for` × 4 + `arith.cmpf` + `select` |
260|258|| **对应 bishengir** | `linalg.pooling_nhwc_max`（Homebrew 不可用，用 affine 替代）|
261|259|| **难度提示** | 需要理解手动循环的 affine.for 语法 |
262|260|
263|261|#### avg_pool_4x4（平均池化）— 下采样
264|262|
265|263|| 维度 | 内容 |
266|264||------|------|
267|265|| **功能** | 2×2 窗口内取平均值，stride=2，4×4 → 2×2 |
268|266|| **AI 角色** | **Average Pooling (平均池化)**：比 max pooling 更平滑的下采样。ResNet 中使用 `avg_pool` 做分类头前的下采样 |
269|267|| **MLIR 模式** | `affine.for` × 4 + 累加 + 除法 |
270|268|| **对应 bishengir** | `linalg.pooling_nchw_sum` + 除法 |
271|269|
272|270|#### global_avg_pool_4x4（全局平均池化）— 分类头
273|271|
274|272|| 维度 | 内容 |
275|273||------|------|
276|274|| **功能** | 对整个特征图求平均，4×4 → 1×1 |
277|275|| **AI 角色** | **Global Average Pooling (全局平均池化, GAP)**：ResNet/MobileNet/GoogleNet 的分类头——代替全连接层，将特征图压缩为类别置信度。参数量为 0，天然防止过拟合 |
278|276|| **MLIR 模式** | `affine.for` × 2 + 累加 + 平均因子 |
279|277|| **难度提示** | 2 层循环比 4 层循环简单，但需要理解 `memref<f32>` 标量存储 |
280|278|
281|279|#### batch_norm_4x4_part1 + part2（批归一化）— 训练稳定性
282|280|
283|281|| 维度 | 内容 |
284|282||------|------|
285|283|| **功能** | Part1: 每个通道求均值 `mean[j] = Σᵢ x[i][j] / N`，Part2: `y = γ × (x - μ) / √(σ² + ε) + β` |
286|284|| **AI 角色** | **Batch Normalization (批归一化, BN)**：CNN 训练的核心技巧——稳定训练、允许更高学习率。ResNet 中每层卷积后都有 BN。LLM 中已被 Layer Norm 替代，但视觉模型仍用 BN |
287|285|| **MLIR 模式** | Part1: `reduction` + `parallel` 混合 iterator；Part2: 5 个 ins 操作数的 `linalg.generic` |
288|286|| **对应 bishengir** | 需拆解为 reduce + broadcast + elemwise 组合 |
289|287|| **难度提示** | ⭐⭐⭐ 最复杂的用例——需要理解多步 pipeline、reduction 与 broadcast 的配合 |
290|288|
291|289|#### layer_norm_4x4（层归一化）— Transformer
292|290|
293|291|| 维度 | 内容 |
294|292||------|------|
295|293|| **功能** | `y = (x - mean) / sqrt(var + eps) * γ + β`，平方差部分 |
296|294|| **AI 角色** | **Layer Normalization (层归一化, LN)**：Transformer 的标配归一化——每个 token 自己做归一化，不依赖 batch 内其他 token。GPT/BERT/LLaMA 每层都有 LN。**比 BN 更适合变长序列** |
297|295|| **MLIR 模式** | `linalg.generic` + `arith.subf` + `arith.mulf` |
298|296|| **对应 bishengir** | 组合模式 |
299|297|| **难度提示** | 概念上相对简单（逐元素），但与 reduce 配合才能完成完整的 LN |
300|298|
301|299|---
302|300|
303|301|## 学习路线建议
304|302|
305|303|```
306|304|初学者 → 先看 ⭐ 8 个 → 建立 linalg.generic 直觉
307|305|    ↓
308|306|有一点点基础 → 看 ⭐⭐ 11 个 → 理解 reduction / broadcast / 组合模式
309|307|    ↓
310|308|掌握 MLIR 核心概念 → 看 ⭐⭐⭐ 9 个 → matmul / conv / 多步 pipeline
311|309|    ↓
312|310|理解 bishengir 降级 → 再跑一遍 variants/compare.sh → 观察 28 个用例的膨胀率
313|311|```
314|312|
315|313|---
316|314|
317|315|## 快速开始
318|316|
319|317|```bash
320|318|export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
321|319|
322|320|# 单个用例
323|321|mlir-opt --convert-linalg-to-affine-loops test-cases/vecadd_128.mlir
324|322|
325|323|# 完整降级到 LLVM
326|324|mlir-opt \
327|325|  --convert-linalg-to-affine-loops \
328|326|  --lower-affine \
329|327|  --convert-scf-to-cf \
330|328|  --convert-func-to-llvm \
331|329|  test-cases/vecadd_128.mlir
332|330|
333|331|# 批量运行
334|332|bash run-demo.sh
335|333|```
336|334|
337|335|---
338|336|
339|337|## 自动化测试
340|338|
341|339|每个 `.mlir` 文件都包含 FileCheck 风格的 `// RUN:` 和 `// CHECK:` 标注，支持两种验证方式：
342|340|
343|341|### 方式 1：使用 run-tests.sh 脚本
344|342|
345|343|```bash
346|344|# 运行全部 28 个测试
347|345|bash run-tests.sh
348|346|
349|347|# 详细模式
350|348|bash run-tests.sh --verbose
351|349|
352|350|# 只运行匹配的用例
353|351|bash run-tests.sh matmul
354|352|```
355|353|
356|354|### 方式 2：使用 LLVM lit + FileCheck
357|355|
358|356|```bash
359|357|# 单个文件
360|358|mlir-opt --convert-linalg-to-affine-loops test-cases/mlir/01_basic/01_vecadd.mlir | FileCheck test-cases/mlir/01_basic/01_vecadd.mlir
361|359|```
362|360|
363|361|### 测试标注说明
364|362|
365|363|- `// RUN:` — 要执行的 mlir-opt 命令（`%s` = 当前文件路径）
366|364|- `// CHECK:` — 验证输出中必须包含的 IR 模式
367|365|- `// CHECK-NOT:` — 验证输出中不能包含的 IR 模式
368|366|
369|367|---
370|368|
371|369|## 矩阵乘法优化方案对比
372|370|
373|371|matmul 的 74× 膨胀源于三重循环完全展开为标量。
374|340|`variants/compare.sh` 直接对比 4 种优化策略：
375|341|
376|342|| Variant | 策略 | LLVM 行数 | vs 基准 | 原理 |
377|343||---------|------|-----------|---------|------|
378|344|| **V0** | 无优化 (基准) | 74 行 | - | 三重循环完全展开 |
379|345|| **V1** | 循环分块 (tile=2x2x1) | 76 行 | +2 行 | 增加 tile 循环层，改善 cache |
380|346|| **V2** | 向量化 (tile+vectorize) | 77 行 | +3 行 | SIMD 指令，减少指令数 |
381|347|| **V3** | **硬件映射 (模拟 mmul)** | **5 行** | **-69 行 (-93%)** | func.call 保留语义，不展开 |
382|348|
383|349|### V3 的核心思路 — bishengir 实际采用的方案
384|350|
385|351|```text
386|352|标准 MLIR 路径 (V0):
387|353|  linalg.matmul → affine.for×3 → scf.for+arith → llvm.load/add/mul/store  (74行)
388|354|
389|355|bishengir 路径 (≈V3):
390|356|  linalg.matmul → hfusion.cube_matmul (1行) → hivm.mmul (1行)
391|357|                                                 ↑
392|358|                                           Ascend NPU Cube 单元
393|359|                                           硬件直接执行矩阵乘
394|360|```
395|361|
396|362|**关键**: 高级操作**保持高级语义**（不展开到标量），直接映射到硬件指令。
397|363|
398|364|#
## Triton 对应代码

每个 MLIR 测试用例对应一个 Triton Python kernel，位于 `test-cases/triton/` 目录下。
目录结构与 mlir/ 一致（`01_basic/` / `02_intermediate/` / `03_advanced/`）。

| 用途 | 命令 |
|------|------|
| 查看映射 | `cat test-cases/triton/MAPPING.md` |
| 跑 Triton (需 NVIDIA GPU) | `python3 test-cases/triton/01_basic/01_vecadd.py` |

详见 `test-cases/triton/README.md`。

## 环境限制
399|365|
400|366|详见 `LIMITATIONS.md`。
401|367|
402|368|**一句话总结**: Homebrew LLVM 22 未编译 Linalg named ops（conv/pooling/fill 等 named 版本），
403|369|但全部可通过 `linalg.generic` 替代（pooling 除外——需 bishengir 自编译版本）。
404|370|bishengir-opt 自编译时包含这些 named op，功能不受影响。
405|371|
406|372|```bash
407|373|bash variants/compare.sh
408|374|```
409|375|
410|376|---
411|377|
412|378|## bishengir ↔ 标准 MLIR 对照
413|379|
414|380|```text
415|381|bishengir:                      标准 MLIR (本 demo):
416|382|────────────────────             ────────────────────
417|383|linalg.generic                  linalg.generic
418|384|    ↓ -convert-linalg-to-hfusion    ↓ --convert-linalg-to-affine-loops
419|385|hfusion.elemwise_binary         affine.for + arith.addf
420|386|    ↓ -convert-hfusion-to-hivm     ↓ --lower-affine --scf-to-cf --func-to-llvm
421|387|hivm.load/vadd/store            llvm.load + llvm.add + llvm.store
422|388|```
423|389|