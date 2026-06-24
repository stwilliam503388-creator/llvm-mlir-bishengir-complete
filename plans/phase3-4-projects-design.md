# Phase 3 & 4 配套项目/代码设计

> 追加到 `plans/phase3-4-detailed-design.md`

---

## 8. 配套代码与示例

### 设计原则

对标 Phase 2 的 `projects/hello-pass/`：每个动手项目必须满足：
- 独立目录，有自己的一键 `run.sh`
- 代码 ≤50 行，初学者能逐行看懂
- 输出明确，能和文档中的"预期输出"对照验证

---

### 项目 1：`projects/mlir-hello/` — MLIR 版 HelloPass

**定位**：Phase 3 的动手入口。学完 `00-从LLVM到MLIR.md` 后，跑通这个项目来"感受" MLIR Pass。

**目录结构**：

```
projects/mlir-hello/
├── CMakeLists.txt        ← 构建配置（~20 行）
├── HelloMLIRPass.cpp     ← 源码（~35 行）
├── test.mlir             ← 测试输入（~15 行）
├── run.sh                ← 一键构建+运行
└── README.md             ← 说明 + 预期输出
```

**源码设计**：

```cpp
// HelloMLIRPass.cpp — 最简 MLIR Pass
// 和 Phase 2 的 HelloPass 结构完全对应

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {
struct HelloMLIRPass : public PassWrapper<HelloMLIRPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // 打印函数名
    func.emitRemark() << "Hello: " << func.getName();
    
    // 统计 Operation 数量
    int opCount = 0;
    func.walk([&](Operation *op) { opCount++; });
    func.emitRemark() << "  Operation 数量: " << opCount;
  }
};
}

// 注册 Pass
std::unique_ptr<Pass> createHelloMLIRPass() {
  return std::make_unique<HelloMLIRPass>();
}
```

**测试输入** (`test.mlir`)：

```mlir
func.func @add(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  func.return %sum : i32
}

func.func @empty() {
  func.return
}
```

**预期输出**：

```
Remark: Hello: add
Remark:   Operation 数量: 3
Remark: Hello: empty
Remark:   Operation 数量: 1
```

**和 Phase 2 HelloPass 的对照**：

| | HelloPass (LLVM) | HelloMLIRPass (MLIR) |
|---|---|---|
| 遍历单位 | `Function &F` | `func::FuncOp` |
| 打印函数名 | `F.getName()` | `func.getName()` |
| 统计指令 | `BB.size()` | `func.walk()` |
| 注册 | `RegisterPass<...> X(...)` | `createHelloMLIRPass()` |
| 运行 | `opt --passes="hello"` | `mlir-opt --pass-pipeline="hello-mlir"` |

**run.sh 设计**：

```bash
#!/bin/bash
set -e
LLVM_PREFIX="/opt/homebrew/opt/llvm"
export PATH="$LLVM_PREFIX/bin:$PATH"

echo "🔨 构建 HelloMLIRPass..."
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(llvm-config --cmakedir)/../mlir"
make -j$(sysctl -n hw.logicalcpu)

echo ""
echo "✅ 构建成功！运行："
mlir-opt --load-pass-plugin ./libHelloMLIRPass.dylib \
         --pass-pipeline="builtin.module(hello-mlir)" \
         ../test.mlir
```

**CMakeLists.txt**：

```cmake
cmake_minimum_required(VERSION 3.20)
project(HelloMLIRPass)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

add_library(HelloMLIRPass MODULE HelloMLIRPass.cpp)
target_link_libraries(HelloMLIRPass PRIVATE MLIR)
```

**文档关联**：`docs/mlir/00-从LLVM到MLIR.md` 末尾引导"→ 动手：[HelloMLIRPass](../../projects/mlir-hello/)"。

---

### 项目 2：`projects/mlir-fusion-demo/` — 算子融合演示

**定位**：学完 `docs/ascend/01-husion-hivm-Dialect详解.md` 后，看一个最小化的融合示例。

**目录结构**：

```
projects/mlir-fusion-demo/
├── input.mlir            ← 融合前的 linalg IR（~20 行）
├── expected.mlir         ← 融合后的 husion IR（~10 行）
├── trace.sh              ← 运行 Lowering + 打印中间 IR
└── README.md             ← 说明 + 对照解读
```

**input.mlir**（两个独立操作：add + mul）：

```mlir
func.func @fused_add_mul(%A: tensor<128xf32>, %B: tensor<128xf32>, %C: tensor<128xf32>)
    -> tensor<128xf32> {
  // 操作 1: add
  %add = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<128xf32>, tensor<128xf32>)
    outs(%A : tensor<128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>

  // 操作 2: mul
  %mul = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%add, %C : tensor<128xf32>, tensor<128xf32>)
    outs(%add : tensor<128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %0 = arith.mulf %a, %b : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>

  func.return %mul : tensor<128xf32>
}
```

**expected.mlir**（融合后，一个 husion 操作）：

```mlir
func.func @fused_add_mul(%A: tensor<128xf32>, %B: tensor<128xf32>, %C: tensor<128xf32>)
    -> tensor<128xf32> {
  %0 = husion.elemwise_binary "add_mul" ins(%A, %B, %C) outs(%A)
      : tensor<128xf32>
  func.return %0 : tensor<128xf32>
}
```

**说明**：30 行 linalg IR → 5 行 husion IR。10 倍压缩。这就是融合的力量。

**注意**：这个项目**不**需要自己能运行（依赖 ascendnpu-ir 构建），它是用来"看"的——配合 `docs/ascend/01` 文档，读者对照 input 和 expected 理解融合做了什么。

---

### 项目 3：`projects/ascend-trace/` — 一条算子走到底

**定位**：学完 Phase 4 全部后，看一个真实算子从 PyTorch 到 NPU 的全程。

**目录结构**：

```
projects/ascend-trace/
├── trace.md              ← 完整的 Lowering 追踪记录
│   ├── Step 1: PyTorch 算子 (Conv2D)
│   ├── Step 2: linalg.generic
│   ├── Step 3: husion.elemwise_binary
│   ├── Step 4: hivm.vadd + hivm.mmad
│   └── Step 5: Ascend 可执行
├── test_conv2d.py        ← Python 测试脚本（如果有可用环境）
└── README.md
```

**trace.md 设计**：

```
## Step 1: PyTorch
torch.nn.Conv2d(3, 64, kernel_size=3)
    ↓ (torch.compile / torch-mlir)

## Step 2: linalg.generic
linalg.conv_2d_nhwc_hwcf {dilations = [1,1], strides = [1,1]}
    ins(%input, %weight) outs(%output)
    ↓ (ConvertLinalgToHusion)

## Step 3: husion
husion.conv2d "forward" ins(%input, %weight) outs(%output)
    : tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>
    ↓ (ConvertHusionToHIVM)

## Step 4: hivm
hivm.load %input_addr : memref<...>
hivm.mmad %w_tile, %x_tile, %acc : vector<16x16xf16>
hivm.store %result, %output_addr : memref<...>
    ↓ (CodeGen)

## Step 5: Ascend NPU 可执行文件
```

**说明**：这是一个"阅读材料"而非可执行项目。让读者看到：你在 PyTorch 里写的一行 `Conv2d`，在编译器后端经历了什么才变成 NPU 指令。

---

## 9. 更新后的文件清单

### 新增动手项目（3 个）

| # | 项目 | 类型 | 文件数 | 定位 |
|---|------|------|--------|------|
| 23 | `projects/mlir-hello/` | 可运行 | 5 | Phase 3 动手入口 |
| 24 | `projects/mlir-fusion-demo/` | 阅读材料 | 3 | Phase 4 融合对照 |
| 25 | `projects/ascend-trace/` | 阅读材料 | 3 | Phase 4 全程追踪 |

### 更新 `projects/README.md`

```markdown
## 已完成项目

| 项目 | 难度 | 对应阶段 | 说明 |
|------|------|---------|------|
| [hello-pass](./hello-pass/) | ⭐ | Phase 2 LLVM | 第一个 LLVM Pass |
| [mlir-hello](./mlir-hello/) | ⭐ | Phase 3 MLIR | MLIR 版 HelloPass |
| [mlir-fusion-demo](./mlir-fusion-demo/) | ⭐⭐ | Phase 4 Ascend | 算子融合对照演示 |
| [ascend-trace](./ascend-trace/) | ⭐⭐ | Phase 4 Ascend | 一条算子走到底 |
```

---

## 10. 总计

| 类别 | 数量 | 说明 |
|------|------|------|
| Phase 3 桥接文档（新写） | 4 篇 | README + 00/01/02 |
| Phase 3 旧文档（包装） | 9 篇 | MLIR-L00~L08 |
| Phase 4 文档（新写） | 5 篇 | README + 00/01/02/03 |
| Phase 3 动手项目 | 1 个 | mlir-hello（可运行） |
| Phase 4 示例项目 | 2 个 | fusion-demo + ascend-trace（阅读材料） |
| 全局更新 | 5 处 | README/SUMMARY/quickstart/projects-README/llvm-03 |

**总共**：25 个文件操作，18 篇文档 + 3 个项目 + 5 处全局更新。
