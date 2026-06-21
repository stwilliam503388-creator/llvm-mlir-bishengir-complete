# (对应 01_basic/07_fill.mlir) ⭐
# 公式: A[i][j] = c, 常量填充
# 一句话: 用常数把数组清空
# 专业角色: 缓冲区初始化, 卷积/矩阵乘前清零输出
# 用在哪: 缓冲区初始化 / 梯度清零
# 降级对比: MLIR linalg.generic+yield, Triton store(val)
# bishengir: linalg.fill (Homebrew 未编译, generic 替代)
import triton
import triton.language as tl
import torch

@triton.jit
def fill_kernel(X, val: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offsets, val)

if __name__ == "__main__":
    x = torch.empty(32, device='cuda')
    fill_kernel[(1,)](x, val=0.0, BLOCK=32)
    assert (x == 0).all(), f"Not all zeros: {x}"
    print(f"✅ Fill: 全部为 {x[0].item()}, sum={x.sum().item()}")