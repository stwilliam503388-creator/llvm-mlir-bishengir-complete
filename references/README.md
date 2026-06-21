# references/ — 参考源码索引

本目录指向 / 说明外部仓库中与本项目相关的源码位置。

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

**对应笔记**: `docs/mlir/MLIR-L06-TritonMLIR体系分析.md`, `MLIR-L07-triton-ascend后端深度分析.md`
