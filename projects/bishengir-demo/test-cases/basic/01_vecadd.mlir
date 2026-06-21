// VecAdd — 向量加法
//
// 功能: C[i] = A[i] + B[i], 逐元素加法
// AI 角色: 残差连接 (Residual Connection)
//   ResNet/Transformer 中每层输出与输入直接相加, 解决深层网络梯度消失.
// 应用场景: ResNet / Transformer / GPT 全系列
// MLIR 模式: linalg.generic + arith.addf, 3行→38行LLVM (12.7×)
// 对应 bishengir: hfusion.elemwise_binary {fun = add}
//
1|// VecAdd — 向量加法
2|//
3|// 功能: C[i] = A[i] + B[i], 逐元素加法
4|// AI 角色: 残差连接 (Residual Connection / Shortcut)
5|//   ResNet/Transformer 中每层输出与输入直接相加, 解决深层网络梯度消失.
6|//   每个 Attention/FFN 层后都有 x + sublayer(x) 的残差连接.
7|// 应用场景: ResNet / Transformer / GPT 全系列, 无处不在
8|// MLIR 模式: linalg.generic + arith.addf, 3行→38行LLVM (12.7×)
9|// 对应 bishengir: hfusion.elemwise_binary {fun = add}
10|//
11|1|
12|2|// VecAdd — 向量加法
13|3|//
14|4|// 功能: C[i] = A[i] + B[i], 逐元素加法
15|5|// AI 角色: 残差连接 (Residual Connection / Shortcut)
16|6|//   ResNet/Transformer 中每层输出与输入直接相加, 解决深层网络梯度消失.
17|7|//   LLM 中每个 Attention/FFN 层后都有 x + sublayer(x) 的残差连接.
18|8|// 应用场景: ResNet / Transformer / GPT 全系列
19|9|// MLIR 模式: linalg.generic + arith.addf, 3行→38行LLVM
20|10|// 对应 bishengir: hfusion.elemwise_binary {fun = add}
21|11|//
22|12|