// Sigmoid — Logistic 激活函数
//
// 功能: sigma(x) = 1 / (1 + e^{-x}), 输出范围 (0, 1)
// AI 角色: 二分类输出层 + RNN 门控. LLaMA 用 sigmoid 做 SwiGLU 门控激活.
// 应用场景: 二分类 / RNN 门控 / SwiGLU 中间步骤
// MLIR 模式: negf + math.exp + addf + divf, 4步组合
// 对应 bishengir: 组合模式可融合为单个 hfusion
//
