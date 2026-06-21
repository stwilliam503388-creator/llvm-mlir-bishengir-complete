// Fill — 张量填充
// 公式: A[i][j] = c, 常量填充
// 一句话: 用常数把数组清空, 准备放新数据
// 专业角色: 缓冲区初始化, 卷积/矩阵乘前清零输出
// 用在哪: 缓冲区初始化 / 梯度清零
// 降级: linalg.generic + yield 常数
// bishengir: linalg.fill (Homebrew 未编译, 用 generic 替代)

module {
  func.func @fill(%A: memref<4x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]} outs(%A : memref<4x4xf32>) {
    ^bb0(%a: f32):
      linalg.yield %c0 : f32
    }
    return
  }
}
