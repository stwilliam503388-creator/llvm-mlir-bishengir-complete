# references/ — 参考源码索引

本目录指向/说明外部仓库中与本项目相关的源码位置。

---

## AscendNPU-IR（官方）

| 项目 | 链接 | 说明 |
|------|------|------|
| **代码仓** | https://github.com/Ascend/AscendNPU-IR | 华为官方 Ascend NPU MLIR 编译器 |
| **文档** | https://ascendnpu-ir.gitcode.com/zh_cn/index.html | 中文文档（GitCode 镜像）|

**本项目的 ascendnpu-ir 与此的关系**：

```
AscendNPU-IR (华为官方)
    └── fork → ascendnpu-ir (Nous Research 维护)
                  └── bishengir (另一个名称, 同一代码库)
```

本项目分析的是 `ascendnpu-ir`（Nous Research fork），它基于华为官方 `AscendNPU-IR` 扩展了自定义 Pass 和 dialect 定义。

**核心目录**：

| 路径 | 说明 |
|------|------|
| `bishengir/include/bishengir/Dialect/` | Dialect 定义（hfusion/hivm）|
| `bishengir/lib/Conversion/` | 转换 Pass：LinalgToHFusion / ArithToHFusion / HFusionToHIVM |
| `bishengir/tools/bishengir-opt/` | 主入口（类似 mlir-opt）|

**对应笔记**: `docs/mlir/L00-速通与bishengir实战.md`, `L08-bishengir-demo可运行流水线.md`

---

## triton-ascend

**位置**: `~/Documents/GitHub-Projects/triton-ascend/`

**核心文件**:

| 路径 | 说明 |
|------|------|
| `python/triton/compiler/compiler.py` | 编译入口 |
| `python/triton/runtime/ascend_interpreter.py` | Ascend 运行时 |
| `include/triton/Dialect/Triton/IR/TritonOps.td` | TT dialect 定义 |
| `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | TritonGPU dialect |
| `lib/Conversion/TritonToTritonGPU/` | 转换 Pass |
| `lib/Conversion/TritonGPUToLLVM/` | LLVM 降级 |

**对应笔记**: `docs/mlir/L06-TritonMLIR体系分析.md`, `L07-triton-ascend后端深度分析.md`
