# Avg Pool — 平均池化 (对应 03_advanced/06_avg_pool.mlir)
# MLIR: affine.for x4 + 累加 + 除法
# Triton: tl.sum / 窗口大小
# 公式: 2x2窗口, stride=2, 取平均值

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
