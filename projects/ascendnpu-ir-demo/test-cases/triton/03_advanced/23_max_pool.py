1|# (对应 mlir/03_advanced/05_max_pool.mlir) ⭐⭐⭐
2|# 公式: 2x2窗口取最大值, stride=2, 4x4->2x2
3|# 一句话: 每4个像素取最亮的1个, 缩小图片
4|# 专业角色: CNN 下采样, 保留最强特征
5|# 用在哪: LeNet/AlexNet/VGG 下采样
6|# 降级对比: MLIR affine.for x4 + cmpf+select (11行), Triton reshape+tl.max
7|# bishengir: linalg.pooling_nhwc_max (需自编译)
8|import triton
9|import triton.language as tl
10|import torch
11|
12|@triton.jit
13|def max_pool_kernel(INPUT, OUTPUT, H, W, KH, KW, BLOCK: tl.constexpr):
14|    pid = tl.program_id(0)
15|    oh = pid // (W // 2)
16|    ow = pid % (W // 2)
17|    kh = tl.arange(0, KH)
18|    kw = tl.arange(0, KW)
19|    vals = tl.load(INPUT + (oh * 2 + kh[:, None]) * W + (ow * 2 + kw[None, :]))
20|    m = tl.max(tl.reshape(vals, (KH * KW,)))
21|    tl.store(OUTPUT + oh * (W // 2) + ow, m)
22|
23|if __name__ == "__main__":
24|    inp = torch.tensor([[1.,5.,2.,3.],[2.,8.,1.,4.],[3.,2.,7.,6.],[0.,1.,5.,9.]], device='cuda')
25|    out = torch.empty(2, 2, device='cuda')
26|    max_pool_kernel[(4,)](inp.flatten(), out.flatten(), 4, 4, 2, 2, BLOCK=4)
27|    print(f"✅ MaxPool: {out.tolist()}")