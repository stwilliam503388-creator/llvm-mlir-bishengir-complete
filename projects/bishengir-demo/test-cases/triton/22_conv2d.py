# (对应 03_advanced/04_conv2d.mlir) ⭐⭐⭐
# 公式: output = sum(输入*核) 滑动窗口
# 一句话: 卷积核在输入上滑动提取特征
# 专业角色: CNN 核心算子, 参数共享+局部连接
# 用在哪: ResNet/VGG/YOLO/CNN 全系
# 降级对比: MLIR generic+reduction x2 (6->85行), Triton 手动窗口
# bishengir: 可被 pattern matching 优化
import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_kernel(INPUT, KERNEL, OUTPUT, H, W, KH, KW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # (oh, ow) 展平
    oh = pid // (W - KW + 1)
    ow = pid % (W - KW + 1)
    kh = tl.arange(0, KH)
    kw = tl.arange(0, KW)
    inp = tl.load(INPUT + (oh + kh[:, None]) * W + (ow + kw[None, :]))
    ker = tl.load(KERNEL + kh[:, None] * KW + kw[None, :])
    out = tl.sum(inp * ker)
    tl.store(OUTPUT + oh * (W - KW + 1) + ow, out)

if __name__ == "__main__":
    input = torch.randn(1, 1, 4, 4, device='cuda')
    kernel = torch.randn(1, 1, 3, 3, device='cuda')
    oh = 4 - 3 + 1
    out = torch.empty(1, 1, oh, oh, device='cuda')
    conv2d_kernel[(oh * oh,)](input.flatten(), kernel.flatten(), out.flatten(), 4, 4, 3, 3, BLOCK=3)
    expected = torch.nn.functional.conv2d(input, kernel)
    max_err = (out - expected).abs().max().item()
    print(f"✅ Conv2D: max_error={max_err:.6f}, output={oh}x{oh}")