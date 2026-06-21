#!/usr/bin/env python3
"""
bishengir-demo.py — 生成 MLIR 测试用例，模拟 bishengir 三阶段降级。

生成三种测试用例，展示 bishengir 的核心功能：
  1. vecadd   → 向量加法（最简，linalg.generic + arith.addf）
  2. matmul   → 矩阵乘法（linalg.matmul）
  3. fused    → 融合 op（add + mul 连续操作，展示融合理念）

每个用例输出到 .mlir 文件，可被 mlir-opt 处理。
"""

from pathlib import Path
import textwrap

OUTPUT_DIR = Path(__file__).parent / "test-cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 用例 1: VecAdd（向量加法）
# ═══════════════════════════════════════════════════════════════════════════
# 对应 bishengir test 文件: linalg-to-hfusion.mlir
# 流程: linalg.generic { arith.addf } → hfusion.elemwise_binary {add} → hivm.vadd
# ═══════════════════════════════════════════════════════════════════════════

VECADD_MLIR = textwrap.dedent("""\
// ==- vecadd.mlir - 向量加法 -==//
//
// bishengir 流水线:
//   输入:  linalg.generic { arith.addf }
//   Pass1: -convert-linalg-to-hfusion  → hfusion.elemwise_binary {add}
//   Pass2: -convert-arith-to-hfusion   → (处理剩余的 arith ops)
//   Pass3: -convert-hfusion-to-hivm    → hivm.load + hivm.vadd + hivm.store
//
// 标准 MLIR 流水线（在当前环境可运行）:
//   --convert-linalg-to-affine-loops  → affine.for + arith.addf
//   --lower-affine                    → scf.for
//   --convert-scf-to-cf + --convert-func-to-llvm → LLVM IR
//--

module {{
  func.func @vecadd(
    %A: memref<{size}xf16>,
    %B: memref<{size}xf16>,
    %C: memref<{size}xf16>
  ) {{
    // 逐元素加法: C[i] = A[i] + B[i]
    linalg.generic {{
      indexing_maps = [
        affine_map<(i) -> (i)>,   // A[i]
        affine_map<(i) -> (i)>,   // B[i]
        affine_map<(i) -> (i)>    // C[i]
      ],
      iterator_types = ["parallel"]
    }} ins(%A, %B : memref<{size}xf16>, memref<{size}xf16>)
      outs(%C : memref<{size}xf16>) {{
    ^bb0(%a: f16, %b: f16, %c: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
    }}
    return
  }}
}}
""")


# ═══════════════════════════════════════════════════════════════════════════
# 用例 2: MatMul（矩阵乘法）
# ═══════════════════════════════════════════════════════════════════════════
# bishengir 中的 linalg.matmul 也会经过类似路径被转换
# ═══════════════════════════════════════════════════════════════════════════

MATMUL_MLIR = textwrap.dedent("""\
// ==- matmul.mlir - 矩阵乘法 -==//
//
// bishengir 中 linalg.matmul 是高阶操作，最终会 lower 到
// HIVM 的 Cube 指令（矩阵乘单元）。
//
// 与 vecadd 的区别:
//   vecadd → hivm.vadd (Vector 单元, 逐元素)
//   matmul → hivm.mmul (Cube 单元, 矩阵乘)
//--

module {{
  func.func @matmul(
    %A: memref<{m}x{k}xf16>,
    %B: memref<{k}x{n}xf16>,
    %C: memref<{m}x{n}xf16>
  ) {{
    // 矩阵乘法: C[m,n] = A[m,k] × B[k,n]
    linalg.matmul ins(%A, %B : memref<{m}x{k}xf16>, memref<{k}x{n}xf16>)
                 outs(%C : memref<{m}x{n}xf16>)
    return
  }}
}}
""")


# ═══════════════════════════════════════════════════════════════════════════
# 用例 3: Fused（融合操作 — add + mul）
# ═══════════════════════════════════════════════════════════════════════════
# 展示 bishengir 的融合能力：连续两个 linalg.generic 可以被融合为
# 一个 hfusion.elemwise_binary 操作。
# ═══════════════════════════════════════════════════════════════════════════

FUSED_MLIR = textwrap.dedent("""\
// ==- fused.mlir - 融合操作 -==//
//
// bishengir 的融合概念:
//   C = A + B
//   D = C * A
//   → 可以通过融合优化为单个 kernel 执行
//
// 标准 MLIR 也会做类似的 fusion:
//   --linalg-fuse-elementwise-ops
// ═══════════════════════════════════════════════════════════════════════════

module {{
  func.func @fused(
    %A: memref<{size}xf32>,
    %B: memref<{size}xf32>,
    %D: memref<{size}xf32>
  ) {{
    // 临时缓冲区
    %C = memref.alloc() : memref<{size}xf32>

    // Step 1: C[i] = A[i] + B[i]
    linalg.generic {{
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"]
    }} ins(%A, %B : memref<{size}xf32>, memref<{size}xf32>)
      outs(%C : memref<{size}xf32>) {{
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }}

    // Step 2: D[i] = C[i] * A[i]
    linalg.generic {{
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"]
    }} ins(%C, %A : memref<{size}xf32>, memref<{size}xf32>)
      outs(%D : memref<{size}xf32>) {{
    ^bb0(%c: f32, %a: f32, %d: f32):
      %prod = arith.mulf %c, %a : f32
      linalg.yield %prod : f32
    }}

    memref.dealloc %C : memref<{size}xf32>
    return
  }}
}}
""")


# ═══════════════════════════════════════════════════════════════════════════
# 用例 4: Conv2D（二维卷积）
# ═══════════════════════════════════════════════════════════════════════════
# 展示更复杂的操作，接近真实深度学习场景
# ═══════════════════════════════════════════════════════════════════════════

CONV2D_MLIR = textwrap.dedent("""\
// ==- conv2d.mlir - 二维卷积 -==//
//
// bishengir 可以处理 linalg.conv_2d_nhwc_hwcf，
// 最终 lower 到 HIVM 的 Cube 指令。
//
// 注意: 标准 mlir-opt 也能处理 linalg.conv
// ═══════════════════════════════════════════════════════════════════════════

module {{
  func.func @conv2d(
    %input: memref<1x{size}x{size}x3xf32>,
    %filter: memref<3x3x3x{channels}xf32>,
    %output: memref<1x{size}x{size}x{channels}xf32>
  ) {{
    linalg.conv_2d_nhwc_hwcf
      {{ dilations = dense<1> : tensor<2xi64>,
         strides = dense<1> : tensor<2xi64> }}
      ins(%input, %filter : memref<1x{size}x{size}x3xf32>,
                           memref<3x3x3x{channels}xf32>)
      outs(%output : memref<1x{size}x{size}x{channels}xf32>)
    return
  }}
}}
""")


def write_file(name: str, content: str):
    """写入 MLIR 文件，返回路径"""
    path = OUTPUT_DIR / name
    path.write_text(content)
    print(f"  ✏️  {path.name}  ({len(content)} chars)")
    return path


def generate():
    """生成所有测试用例"""
    print("╔═══════════════════════════════════════╗")
    print("║  bishengir-demo: 生成测试用例         ║")
    print("╚═══════════════════════════════════════╝")
    print()

    # VecAdd — 小、中、大三组参数
    for size in [128, 1024, 4096]:
        write_file(f"vecadd_{size}.mlir",
                    VECADD_MLIR.format(size=size))

    # MatMul — 两组参数
    for m, k, n in [(4, 4, 4), (128, 64, 128)]:
        write_file(f"matmul_{m}x{k}x{n}.mlir",
                    MATMUL_MLIR.format(m=m, k=k, n=n))

    # Fused — 两组
    for size in [128, 1024]:
        write_file(f"fused_{size}.mlir",
                    FUSED_MLIR.format(size=size))

    # Conv2D — 一组
    write_file("conv2d_small.mlir",
               CONV2D_MLIR.format(size=8, channels=16))

    print()
    print(f"  📂 输出目录: {OUTPUT_DIR}")
    print(f"  📊 共 {len(list(OUTPUT_DIR.glob('*.mlir')))} 个文件")


if __name__ == "__main__":
    generate()
