1|# (对应 mlir/01_basic/07_fill.mlir) ⭐
2|# 公式: A[i][j] = c, 常量填充
3|# 一句话: 用常数把数组清空
4|# 专业角色: 缓冲区初始化, 卷积/矩阵乘前清零输出
5|# 用在哪: 缓冲区初始化 / 梯度清零
6|# 降级对比: MLIR linalg.generic+yield, Triton store(val)
7|# bishengir: linalg.fill (Homebrew 未编译, generic 替代)
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def fill_kernel(X, val: tl.constexpr, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    offsets = pid * BLOCK + tl.arange(0, BLOCK)
16|    tl.store(X + offsets, val)
17|
18|if __name__ == "__main__":
19|    x = torch.empty(32, device='cuda')
20|    fill_kernel[(1,)](x, val=0.0, BLOCK=32)
21|    assert (x == 0).all(), f"Not all zeros: {x}"
22|    print(f"✅ Fill: 全部为 {x[0].item()}, sum={x.sum().item()}")