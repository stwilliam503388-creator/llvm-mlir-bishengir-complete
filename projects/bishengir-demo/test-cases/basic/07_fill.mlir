// Fill — 张量填充 (模拟 linalg.fill)
//
// 功能: 用常数值填充整个张量
// AI 角色: 初始化缓冲区 — 卷积/矩阵乘前清零输出, 梯度清零
// 应用场景: 所有需要预先零初始化的情况
// MLIR 模式: linalg.generic + yield %cst
// 对应 bishengir: linalg.fill (Homebrew 未编译, 用 generic 替代)
//
1|// ==- fill_4x4.mlir - 张量填充（模拟 linalg.fill）-==//
2|// 用 linalg.generic 实现 fill 语义
3|// 原因: Homebrew LLVM 22 未编译 linalg.fill named op
4|// 降级: 4 行 → 39 行 LLVM
5|
6|module {
7|  func.func @fill(%A: memref<4x4xf32>) {
8|    %c0 = arith.constant 0.0 : f32
9|    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} outs(%A : memref<4x4xf32>) {
10|    ^bb0(%a: f32):
11|      linalg.yield %c0 : f32
12|    }
13|    return
14|  }
15|}
16|