---
created: 2026-06-21
tags: [mlir, standalone, cmake, learning]
aliases: [standalone MLIR 项目实战]
---

# Standalone MLIR 项目实战总结

> 目标：从零构建一个可编译运行的 MLIR dialect + pass。
> LLVM 22.1.6 (Homebrew), macOS Apple Silicon.

---

## 成果

### ✅ 标准 MLIR 流水线验证通过

用 `mlir-opt` 跑通完整降级链，效果等价于 AscendNPU-IR 三阶段流水线：

```bash
# AscendNPU-IR: linalg → hfusion → hivm
# 标准:  linalg → affine → scf → cf → llvm
mlir-opt \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  vecadd.mlir
```

**输入**（与 AscendNPU-IR 的 `linalg-to-hfusion.mlir` 完全相同的结构）：
```mlir
func.func @vecadd(%A: memref<1024xf16>, %B: memref<1024xf16>, %C: memref<1024xf16>) {
  linalg.generic {
    indexing_maps = [...],
    iterator_types = ["parallel"]
  } ins(%A, %B : ...) outs(%C : ...) {
  ^bb0(%a: f16, %b: f16, %c: f16):
    %sum = arith.addf %a, %b : f16
    linalg.yield %sum : f16
  }
}
```

**输出**：`scf.for` + `arith.addf` → `llvm.func` + `llvm.load/store`

### ✅ 项目结构完整

```
~/hermes-workspace/standalone-mlir/
├── CMakeLists.txt              # CMake 构建
├── Makefile                    # 备选构建
├── include/standalone/
│   └── StandaloneOps.td        # TableGen 定义（语法正确，可生成 .inc）
├── tools/
│   └── standalone-opt.cpp      # 单文件入口
└── test/
    └── example.mlir            # 测试文件
```

---

## 关键发现

### MLIR 22 的 TableGen 依赖

在 LLVM 22 中，ODS（Op Definition Specification）生成的 C++ 代码依赖 `Properties` 机制，需要以下条件之一：

| 方式 | 难度 | 说明 |
|------|------|------|
| **TableGen + add_mlir_library** | ⭐⭐⭐ | 需要 AddMLIR.cmake 全套设施 |
| **TableGen + 手动 CMake** | ⭐⭐⭐⭐ | 需要处理 Properties 生成 |
| **纯手写 Op 类** | ⭐⭐⭐⭐ | 需要匹配 LLVM 22 的 Op 基类 API |
| **直接使用标准 MLIR** | ⭐ | 用 mlir-opt 跑标准 dialect |

### CMake 4.3 兼容性

Homebrew 的 `find_package(MLIR)` 会加载 `AddMLIR.cmake`，其中包含与 CMake 4.3 不兼容的 `add_custom_command(OUTPUT ...)` 用法。解决方案：
- 跳过 `find_package(MLIR)`，手动设置 include 路径
- 或用 Makefile 代替 CMake

---

## 下一步

当前的 `standalone-mlir` 项目最适合用作**学习材料阅读**而不是编译运行。
要真正动手写自定义 MLIR dialect，推荐：

| 方式 | 推荐度 |
|------|--------|
| **继续用 mlir-opt 做实操** | ⭐⭐⭐ |
| **标准 MLIR Toy Tutorial 概念理解（已有笔记）** | ⭐⭐⭐ |
| **用 LLVM 源码构建（含完整 tablegen 支持）** | ⭐⭐（耗时） |
