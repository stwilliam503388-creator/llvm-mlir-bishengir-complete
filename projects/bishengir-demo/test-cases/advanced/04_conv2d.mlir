// Conv2D — 二维卷积 (valid padding)
//
// 功能: 输入4×4+核3×3→输出2×2, 标准卷积层
// AI 角色: CNN 核心算子 — ResNet/VGG/YOLO 全部依赖卷积提取特征.
//   多模态模型 (GPT-4V/CLIP) 的视觉编码器仍用卷积.
// 应用场景: CNN 全系 (ResNet/VGG/YOLO)
// MLIR 模式: linalg.generic + reduction × 2, 6行→85行LLVM
// 对应 bishengir: 可被 pattern matching 优化
//
1|// ==- conv2d_4x4.mlir - 二维卷积（valid padding）-==//
2|// 用 linalg.generic 实现 2D 有效卷积
3|// 输入 4x4, 卷积核 3x3, 输出 2x2
4|// 原因: Homebrew LLVM 22 未编译 linalg.conv_2d named op
5|// 对应 AscendNPU-IR: bishengir 通过 linalg.generic 模式匹配
6|// 降级: 6 行 → 85 行 LLVM
7|
8|module {
9|  func.func @conv2d(%input: memref<4x4xf32>, %kernel: memref<3x3xf32>, %output: memref<2x2xf32>) {
10|    linalg.generic {indexing_maps = [affine_map<(oh,ow,kh,kw) -> (oh+kh, ow+kw)>, affine_map<(oh,ow,kh,kw) -> (kh, kw)>, affine_map<(oh,ow,kh,kw) -> (oh, ow)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%input, %kernel : memref<4x4xf32>, memref<3x3xf32>) outs(%output : memref<2x2xf32>) {
11|    ^bb0(%a: f32, %k: f32, %b: f32):
12|      %prod = arith.mulf %a, %k : f32
13|      %sum = arith.addf %b, %prod : f32
14|      linalg.yield %sum : f32
15|    }
16|    return
17|  }
18|}
19|