// Fused — 算子融合演示 (add + mul)
//
// 功能: C = A + B; D = C * A, 连续两个逐元素操作
// AI 角色: 算子融合概念演示 (Kernel Fusion)
//   编译器将两个连续 kernel 合并为一个, 减少内存读写, 提升利用率.
// 应用场景: 编译器优化概念演示
// MLIR 模式: 连续两次 linalg.generic
// 对应 bishengir: HFusion 算子融合概念演示
//
1|// ==- fused_128.mlir - 融合操作（bishengir demo）-==//
2|//
3|// 对应 AscendNPU-IR 源码:
4|//   本用例在官方测试中没有直接对应，演示 bishengir 的融合概念。
5|//   bishengir 中的融合等价:
6|//     linalg-fuse-elementwise-ops (若支持)
7|//     将连续两个 linalg.generic 合并为一个 hfusion.elemwise_binary
8|//
9|// C = A + B; D = C * A  →  融合后: D[i] = (A[i] + B[i]) * A[i]
10|// 融合减少一次内存读写，是 bishengir 的核心优化能力之一.
11|//===
12|
13|module {
14|  func.func @fused(%A: memref<128xf32>, %B: memref<128xf32>, %D: memref<128xf32>) {
15|    %C = memref.alloc() : memref<128xf32>
16|    linalg.generic {
17|      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
18|      iterator_types = ["parallel"]
19|    } ins(%A, %B : memref<128xf32>, memref<128xf32>)
20|      outs(%C : memref<128xf32>) {
21|    ^bb0(%a: f32, %b: f32, %c: f32):
22|      %sum = arith.addf %a, %b : f32
23|      linalg.yield %sum : f32
24|    }
25|    linalg.generic {
26|      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
27|      iterator_types = ["parallel"]
28|    } ins(%C, %A : memref<128xf32>, memref<128xf32>)
29|      outs(%D : memref<128xf32>) {
30|    ^bb0(%c: f32, %a: f32, %d: f32):
31|      %prod = arith.mulf %c, %a : f32
32|      linalg.yield %prod : f32
33|    }
34|    memref.dealloc %C : memref<128xf32>
35|    return
36|  }
37|}
38|