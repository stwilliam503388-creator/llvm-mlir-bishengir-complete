1|# (对应 mlir/03_advanced/04_conv2d.mlir) ⭐⭐⭐
2|# 公式: output = sum(输入*核) 滑动窗口
3|# 一句话: 卷积核在输入上滑动提取特征
4|# 专业角色: CNN 核心算子, 参数共享+局部连接
5|# 用在哪: ResNet/VGG/YOLO/CNN 全系
6|# 降级对比: MLIR generic+reduction x2 (6->85行), Triton 手动窗口
7|# bishengir: 可被 pattern matching 优化
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def conv2d_kernel(INPUT, KERNEL, OUTPUT, H, W, KH, KW, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)  # (oh, ow) 展平
15|    oh = pid // (W - KW + 1)
16|    ow = pid % (W - KW + 1)
17|    kh = tl.arange(0, KH)
18|    kw = tl.arange(0, KW)
19|    inp = tl.load(INPUT + (oh + kh[:, None]) * W + (ow + kw[None, :]))
20|    ker = tl.load(KERNEL + kh[:, None] * KW + kw[None, :])
21|    out = tl.sum(inp * ker)
22|    tl.store(OUTPUT + oh * (W - KW + 1) + ow, out)
23|
24|if __name__ == "__main__":
25|    input = torch.randn(1, 1, 4, 4, device='cuda')
26|    kernel = torch.randn(1, 1, 3, 3, device='cuda')
27|    oh = 4 - 3 + 1
28|    out = torch.empty(1, 1, oh, oh, device='cuda')
29|    conv2d_kernel[(oh * oh,)](input.flatten(), kernel.flatten(), out.flatten(), 4, 4, 3, 3, BLOCK=3)
30|    expected = torch.nn.functional.conv2d(input, kernel)
31|    max_err = (out - expected).abs().max().item()
32|    print(f"✅ Conv2D: max_error={max_err:.6f}, output={oh}x{oh}")