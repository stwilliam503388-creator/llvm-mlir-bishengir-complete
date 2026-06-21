# standalone-mlir — 从零构建 MLIR dialect

从 TableGen 定义到可运行的 `standalone-opt`，完整展示如何创建 MLIR dialect。

## 结构

```
standalone-mlir/
├── CMakeLists.txt                              — CMake 构建
├── Makefile                                    — 备选 Makefile
├── include/standalone/
│   ├── StandaloneOps.td                        — TableGen dialect + 6 ops
│   └── StandaloneDialect.h                     — C++ dialect 头文件
├── tools/
│   └── standalone-opt.cpp                      — 全合一入口（dialect + 2 passes + main）
└── test/
    └── example.mlir                            — 测试输入
```

## Dialect: `standalone`

| Op | 功能 |
|----|------|
| `standalone.constant` | 常量张量 |
| `standalone.add` | 逐元素加法 |
| `standalone.mul` | 逐元素乘法 |
| `standalone.transpose` | 矩阵转置 |
| `standalone.print` | 打印张量 |
| `standalone.return` | 函数返回 |

## Passes

| Pass | 类型 | 功能 |
|------|------|------|
| `-count-ops` | 分析 Pass | 统计 module 中各 op 出现次数 |
| `-elim-transpose` | 转换 Pass | 检测冗余 transpose 并消除 |

## 构建

```bash
cd projects/standalone-mlir

# 方式一: CMake
export MLIR_DIR="/opt/homebrew/opt/llvm/lib/cmake/mlir"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 方式二: Makefile
make -C build

# 运行
./build/standalone-opt test/example.mlir
./build/standalone-opt test/example.mlir --count-ops
```

## 注意事项

- LLVM 22 的 ODS 生成的 `.h.inc` 只有 forward declaration，完整类定义需要通过 `#define GET_OP_CLASSES` 获取
- 跳过 `find_package(MLIR)` 以避免 `AddMLIR.cmake` 与 cmake 4.3 的兼容问题
- mlir-tblgen 已验证成功，完整 C++ 编译需要 LLVM 22 的 Properties 机制

## 对照

| 组件 | 本项目 | bishengir (ascendnpu-ir) |
|------|--------|-------------------------|
| Dialect 定义 | TableGen | TableGen |
| Pass 注册 | `registerPasses()` | `InitAllPasses.h` |
| 主工具 | `standalone-opt` | `bishengir-opt` |
