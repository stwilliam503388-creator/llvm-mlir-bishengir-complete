# (对应 03_advanced/05_max_pool.mlir) ⭐⭐⭐
# 公式: 2x2窗口取最大值, stride=2, 4x4->2x2
# 一句话: 每4个像素取最亮的1个, 缩小图片
# 专业角色: CNN 下采样, 保留最强特征
# 用在哪: LeNet/AlexNet/VGG 下采样
# 降级对比: MLIR affine.for x4 + cmpf+select (11行), Triton reshape+tl.max
# bishengir: linalg.pooling_nhwc_max (需自编译)
import triton
import triton.language as tl
import torch

@triton.jit
def max_pool_kernel(INPUT, OUTPUT, H, W, KH, KW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid // (W // 2)
    ow = pid % (W // 2)
    kh = tl.arange(0, KH)
    kw = tl.arange(0, KW)
    vals = tl.load(INPUT + (oh * 2 + kh[:, None]) * W + (ow * 2 + kw[None, :]))
    m = tl.max(tl.reshape(vals, (KH * KW,)))
    tl.store(OUTPUT + oh * (W // 2) + ow, m)

if __name__ == "__main__":
    inp = torch.tensor([[1.,5.,2.,3.],[2.,8.,1.,4.],[3.,2.,7.,6.],[0.,1.,5.,9.]], device='cuda')
    out = torch.empty(2, 2, device='cuda')
    max_pool_kernel[(4,)](inp.flatten(), out.flatten(), 4, 4, 2, 2, BLOCK=4)
    print(f"✅ MaxPool: {out.tolist()}")