# (对应 01_basic/03_tanh.mlir) ⭐
# 公式: y = tanh(x), 输出 (-1, 1)
# 一句话: 把任意数压缩到 -1~1 之间
# 专业角色: RNN/LSTM 门控信号, GELU 近似用到此函数
# 用在哪: RNN/LSTM / GELU 中间步骤
# 降级对比: MLIR math.tanh, Triton tl.tanh(x)
# bishengir: hfusion.elemwise_unary {fun = tanh}
import triton
import triton.language as tl
import torch

@triton.jit
def tanh_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.tanh(x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.linspace(-5, 5, 128, device='cuda')
    y = torch.empty(128, device='cuda')
    tanh_kernel[(4,)](x, y, BLOCK=32)
    torch.testing.assert_close(y, torch.tanh(x), atol=1e-5, rtol=1e-5)
    print(f"✅ Tanh: min={y.min().item():.4f}, max={y.max().item():.4f}")