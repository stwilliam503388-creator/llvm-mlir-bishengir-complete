// Global Avg Pool — 全局平均池化
//
// 功能: 对整个特征图求平均, 4×4→1×1
// AI 角色: CNN 分类头前的最后一层 — 代替 Flatten+FC, 参数量为 0.
//   ResNet/MobileNet/GoogleNet 经典设计模式.
// 应用场景: ResNet/MobileNet/GoogleNet 分类头
// MLIR 模式: affine.for ×2 + 累加 + 平均因子
// 对应 bishengir: 需组合 reduce + 除法
//
