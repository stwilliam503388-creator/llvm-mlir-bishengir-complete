# LLVM 22 环境限制说明

本项目的测试用例在 **Homebrew LLVM 22.1.6** 环境下验证。
Homebrew 打包的 LLVM 编译时**没有包含全部 MLIR 特性**，导致部分 named op 不可用。

## 不可用的 named op 及原因

| Op | 状态 | 原因 | Homebrew LLVM | bishengir 自编译 |
|----|------|------|---------------|-----------------|
| `linalg.fill` | ✅ 已通过 generic 替代 | named op 未编译入 Homebrew | 不可用 | 可用 |
| `linalg.conv_2d_nhwc_hwcf` | ✅ 已通过 generic 替代 | named op 未编译入 Homebrew | 不可用 | 可用 |
| `linalg.conv_2d_nchw_fchw` | ✅ 已通过 generic 替代 | named op 未编译入 Homebrew | 不可用 | 可用 |
| `linalg.pooling_nhwc_max` | ❌ 无法修复 | 非可逆 indexing map | 不可用 | 可用 |
| `linalg.pooling_nchw_max` | ❌ 无法修复 | 非可逆 indexing map | 不可用 | 可用 |
| `linalg.pooling_nchw_sum` | ❌ 无法修复 | 非可逆 indexing map | 不可用 | 可用 |

## 不可修复的根本原因

`linalg.generic` 要求 indexing maps 是**可逆的**（bijective）。而 pooling 的 stride > 1 导致多个输入位置映射到同一个输出位置——无法用 `linalg.generic` 表达。

```text
# Max Pool, kernel=2, stride=2:
# 输入 4x4 的 (0,0), (0,1), (1,0), (1,1) 都映射到输出 (0,0)
# → 非可逆映射 → linalg.generic 拒绝
```

这需要 named op（`linalg.pooling_nchw_max`）或手动 `affine.for` 编写。
bishengir 自编译的版本包含这些 named op，所以 bishengir-opt 可以处理 pooling。

## 如果需要在当前环境测试 pooling

可以使用 `mlir-opt` 的 affine dialect 手动编写：

```mlir
affine.for %oh = 0 to 2 {
  affine.for %ow = 0 to 2 {
    // 在每个 2x2 窗口内求最大值
    affine.for %kh = 0 to 2 {
      affine.for %kw = 0 to 2 {
        // 比较 %input[oh*2+kh][ow*2+kw]
      }
    }
  }
}
```

但这种方式跳过了 `linalg.generic` 的优化机会，不推荐作为测试用例。

## 汇总

| Op | Homebrew mlir-opt | bishengir-opt (自编译) |
|----|------------------|----------------------|
| 所有 linalg.generic 操作 | ✅ | ✅ |
| linalg.matmul | ✅ | ✅ |
| 所有 arith op | ✅ | ✅ |
| linalg.fill (named) | ❌ → generic 替代 ✅ | ✅ |
| linalg.conv_2d (named) | ❌ → generic 替代 ✅ | ✅ |
| linalg.pooling (named) | ❌ | ✅ |
