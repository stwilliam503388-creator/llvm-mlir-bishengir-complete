# (对应 03_advanced/06_avg_pool.mlir) ⭐⭐⭐
# 公式: 2x2窗口取平均值, stride=2, 4x4->2x2
# 一句话: 取窗口平均值, 比 max pool 更平滑
# 专业角色: CNN 下采样, ResNet 分类头用全局平均池化
# 用在哪: ResNet 分类头 / 平滑下采样
# 降级对比: MLIR affine.for x4 + 累加+除法, Triton reshape+tl.sum/4
# bishengir: linalg.pooling_nchw_sum (需自编译)
import triton
import triton.language as tl
import torch

@triton.jit
def avg_pool_kernel(INPUT, OUTPUT, H, W, KH, KW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid // (W // 2)
    ow = pid % (W // 2)
    kh = tl.arange(0, KH)
    kw = tl.arange(0, KW)
    vals = tl.load(INPUT + (oh * 2 + kh[:, None]) * W + (ow * 2 + kw[None, :]))
    s = tl.sum(tl.reshape(vals, (KH * KW,))) / (KH * KW)
    tl.store(OUTPUT + oh * (W // 2) + ow, s)

if __name__ == "__main__":
    inp = torch.tensor([[1.,5.,2.,3.],[2.,8.,1.,4.],[3.,2.,7.,6.],[0.,1.,5.,9.]], device='cuda')
    out = torch.empty(2, 2, device='cuda')
    avg_pool_kernel[(4,)](inp.flatten(), out.flatten(), 4, 4, 2, 2, BLOCK=4)
    print(f"✅ AvgPool: {out.tolist()}")