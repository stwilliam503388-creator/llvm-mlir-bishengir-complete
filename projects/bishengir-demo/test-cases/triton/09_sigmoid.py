# (对应 02_intermediate/01_sigmoid.mlir) ⭐⭐
# 公式: sigma(x) = 1 / (1 + e^{-x}), 输出 (0, 1)
# 一句话: 把任意数压缩到 0~1 之间
# 专业角色: 二分类输出层, RNN 门控, SwiGLU (LLaMA) 用 sigmoid 做门控
# 用在哪: 二分类 / RNN门控 / SwiGLU 中间步骤
# 降级对比: MLIR 4步 (negf+exp+addf+divf), Triton 1步 tl.sigmoid(x)
# bishengir: 组合后可融合
import triton
import triton.language as tl
import torch

@triton.jit
def sigmoid_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.sigmoid(x)
    tl.store(Y + offsets, y)

if __name__ == "__main__":
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device='cuda')
    y = torch.empty(5, device='cuda')
    sigmoid_kernel[(1,)](x, y, BLOCK=32)
    print(f"✅ Sigmoid: {dict(zip(x.cpu().tolist(), [f'{v:.4f}' for v in y.cpu().tolist()]))}")