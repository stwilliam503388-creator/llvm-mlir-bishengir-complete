# Step 1 & Step 2 实施计划

> Status: executed | Created: 2026-06-21 | PR: https://github.com/stwilliam503388-creator/ascend-npu-compiler-learning/pull/13
> 目标：让项目"可运行" + 填平 Primer 到 LLVM 的断层
> 当前仓库 HEAD: d9c91968

---

## 环境基线

| 项目 | 值 |
|------|-----|
| OS | macOS 26.5.1 (Apple Silicon) |
| LLVM | 22.1.6 (Homebrew keg-only) |
| LLVM 路径 | `/opt/homebrew/opt/llvm/` |
| cmake dir | `/opt/homebrew/Cellar/llvm/22.1.6/lib/cmake/llvm` |
| C++ 标准 | c++17 |

---

## Step 1: 让项目"可运行"（2 个交付物）

### 任务 1.1: 环境搭建指南

**文件**: `docs/llvm/00-环境搭建.md`

**目标**: 零基础读者跟着做完后能运行 `opt --version` 并看到版本号

**内容大纲**:
```
1. macOS (Apple Silicon)
   1.1 brew install llvm (keg-only 说明)
   1.2 添加到 PATH (/opt/homebrew/opt/llvm/bin)
   1.3 验证: llvm-config --version → 22.1.6
   1.4 验证: opt --version → LLVM 22.1.6

2. Linux (Ubuntu/Debian)
   2.1 apt install llvm-18-dev
   2.2 或使用 llvm.sh 脚本安装最新版
   2.3 验证

3. 常见坑
   3.1 macOS keg-only: brew 不自动加 PATH，必须手动 export
   3.2 Linux 版本过旧: apt 默认可能是 llvm-14，不满足需求
   3.3 cmake 找不到 LLVM: 需要设置 CMAKE_PREFIX_PATH
```

**验证**: 新读者照着做 15 分钟内配好环境

---

### 任务 1.2: Hello Pass 项目

**目录**: `projects/hello-pass/`

**文件 1**: `projects/hello-pass/CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.20)
project(HelloPass)

# 查找 LLVM（支持 Homebrew keg-only 路径）
set(CMAKE_CXX_STANDARD 17)
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 构建共享库 Pass
add_library(HelloPass MODULE HelloPass.cpp)

# 不链接所有 LLVM 库，仅链接需要的（减小体积）
target_link_libraries(HelloPass LLVM)
```

**文件 2**: `projects/hello-pass/HelloPass.cpp`
```cpp
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {
  // HelloPass: 最简单的 FunctionPass
  // 遍历每个函数，打印函数名
  struct HelloPass : public FunctionPass {
    static char ID;
    HelloPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      errs() << "Hello: " << F.getName() << "\n";

      // 打印函数参数数量
      errs() << "  参数数量: " << F.arg_size() << "\n";

      // 打印基本块数量
      errs() << "  基本块数量: " << F.size() << "\n";

      // 返回 false 表示没有修改 IR
      return false;
    }
  };
}

char HelloPass::ID = 0;

// 注册 Pass — opt 通过 -hello 加载
static RegisterPass<HelloPass> X("hello", "Hello World Pass",
                                  false /* 不修改 CFG */,
                                  false /* 分析 Pass */);
```

**文件 3**: `projects/hello-pass/test.ll`
```llvm
; 测试输入: 一个简单的 LLVM IR 函数
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

define void @say_hello() {
entry:
  ret void
}
```

**文件 4**: `projects/hello-pass/run.sh`
```bash
#!/bin/bash
set -e

# macOS: Homebrew keg-only LLVM 路径
LLVM_PREFIX="/opt/homebrew/opt/llvm"
export PATH="$LLVM_PREFIX/bin:$PATH"

# 如果 llvm-config 不在 PATH，尝试找
if ! command -v llvm-config &>/dev/null; then
  echo "❌ 找不到 llvm-config。请先安装 LLVM："
  echo "   brew install llvm"
  echo "   然后 export PATH=\"/opt/homebrew/opt/llvm/bin:\$PATH\""
  exit 1
fi

echo "🔨 构建 HelloPass..."
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(llvm-config --cmakedir)"
make -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo ""
echo "✅ 构建成功！运行测试："
echo ""

# 运行 Pass
opt -load-pass-plugin ./libHelloPass.dylib \
    --passes="hello" \
    ../test.ll \
    -disable-output

echo ""
echo "---"
echo "🎉 HelloPass 运行成功！"
echo ""
echo "如果要看 IR 变换结果（此 Pass 不修改 IR）："
echo "  opt -load-pass-plugin ./libHelloPass.dylib --passes=\"hello\" ../test.ll -S"
```

**文件 5**: `projects/hello-pass/README.md`
```markdown
# HelloPass — 你的第一个 LLVM Pass

本 Pass 是最简单的 LLVM FunctionPass：
- 遍历每个函数
- 打印函数名、参数数量、基本块数量
- 不修改 IR

## 快速开始

```bash
chmod +x run.sh
./run.sh
```

## 预期输出

```
Hello: add
  参数数量: 2
  基本块数量: 1
Hello: say_hello
  参数数量: 0
  基本块数量: 1
```

## 文件说明

| 文件 | 作用 |
|------|------|
| CMakeLists.txt | 构建配置 |
| HelloPass.cpp | Pass 实现（30行） |
| test.ll | 测试输入 |
| run.sh | 一键构建+运行 |
```

**验证标准**:
- [ ] 在 macOS 上 `chmod +x run.sh && ./run.sh` 能成功
- [ ] 输出 "Hello: add" 和 "Hello: say_hello"
- [ ] Linux 用户只需改 run.sh 中的 LLVM_PREFIX

---

### 任务 1.3: 更新顶层索引

**文件**: `projects/README.md`（已有，需更新）

**修改**: 在项目列表中添加 hello-pass 条目
```markdown
### 已完成的动手项目

| 项目 | 难度 | 对应阶段 | 说明 |
|------|------|---------|------|
| [hello-pass](./hello-pass/) | ⭐ | Phase 2 LLVM | 第一个 LLVM Pass：打印函数信息 |
```

---

## Step 2: 填平 Primer 到 LLVM 的断层（3 个交付物）

### 任务 2.1: LLVM IR 快速入门

**文件**: `docs/llvm/01-LLVM-IR快速入门.md`

**目标**: 读者看完后能：
1. 把一个简单的 C 函数"翻译"成 LLVM IR 形式
2. 用 `clang -S -emit-llvm` 自己生成 IR
3. 读懂 SSA、基本块、phi 节点三个核心概念

**内容大纲**:
```
1. 从 Primer 出发: IR 在编译流水线中的位置
   - 回顾 primer/01-AST与IR.md 中的"蓝图纸"比喻
   - LLVM IR 就是编译器世界通用的"蓝图纸语言"

2. 第一眼: 一个加法函数的 IR
   - C: int add(int a, int b) { return a + b; }
   - 用 clang -S -emit-llvm 生成 .ll
   - 逐行解读: define、entry、%变量、ret

3. 三个核心概念 (每个概念配独立小例子)
   - SSA (Static Single Assignment): 每个变量只赋值一次
     比喻: 流水线上的每个工位只加工一次就传下去
   - 基本块 (Basic Block): 没有分支的直线代码段
     比喻: 高速公路的一个直线路段，入口和出口之间不能变道
   - phi 节点: 多个前驱基本块的汇合点
     比喻: 立交桥合流处，车从哪条路来就选哪个值

4. 动手: 自己生成 IR
   - 写一个 if-else 的 C 函数
   - clang -S -emit-llvm 生成 IR
   - 找到 IR 中的基本块和 phi 节点

5. 小结: LLVM IR 学了能干什么?
   - 看懂编译器中间输出
   - 为写 Pass 做准备（下一章）
```

**验证标准**:
- [ ] 包含至少 3 个代码示例（C → LLVM IR 对照）
- [ ] SSA/基本块/phi 三个概念有独立比喻
- [ ] 读者可以自己生成并阅读 IR
- [ ] 不超过 250 行（初学者消化能力有限）

---

### 任务 2.2: 第一个 LLVM Pass（教学版）

**文件**: `docs/llvm/02-第一个LLVM-Pass.md`

**目标**: 读者看完后能：
1. 理解 Pass 的本质："遍历 IR，做点什么"
2. 自己修改 HelloPass 添加一个新功能
3. 理解 FunctionPass 的生命周期

**内容大纲**:
```
1. 回顾: Pass 在编译流水线中的角色
   - 连接 primer/02-Pass与Lowering.md
   - "质检员"比喻: 每个 Pass 检查/优化 IR 的一个方面

2. 从 HelloPass 出发: 逐行解读
   - #include 头文件是干什么的
   - struct HelloPass : public FunctionPass 继承关系
   - runOnFunction() — 每个函数被调用一次
   - errs() — LLVM 的"打印到终端"
   - return false — 我没改 IR

3. 动手修改: 给 HelloPass 加一个新功能
   - 挑战1: 打印每个基本块的指令数量
   - 挑战2: 统计函数中 add 指令的数量
   - 挑战3: 找出参数最多的函数

4. Pass 的类型简介
   - FunctionPass: 按函数遍历（最常用）
   - ModulePass: 按模块遍历
   - BasicBlockPass: 按基本块遍历
   - 不深入，只需知道"有这几种"

5. 小结与下一步
```

**代码片段（挑战1的参考答案）**:
```cpp
// 在 runOnFunction 中添加:
for (auto &BB : F) {
  errs() << "  基本块 " << BB.getName() 
         << " 有 " << BB.size() << " 条指令\n";
}
```

**验证标准**:
- [ ] 链接到 projects/hello-pass/ 的实际代码
- [ ] 包含 3 个动手挑战
- [ ] 每个挑战有预期输出示例
- [ ] 不超过 200 行

---

### 任务 2.3: LLVM 生态速览（可选的精简版）

**文件**: `docs/llvm/03-LLVM工具箱速览.md`

**目标**: 读完知道 LLVM 有哪些工具，各干什么

**内容大纲**:
```
1. 你不需要全部学完才开始
   - 只需要 5 个工具就够写 Pass

2. 核心工具速查表

| 工具 | 一句话 | 什么时候用 |
|------|--------|-----------|
| clang | C/C++ 编译器前端 | 把 .c 变成 .ll |
| opt | IR 优化器/Pass 运行器 | 加载你的 Pass 跑在 IR 上 |
| llc | IR → 汇编 | 看最终生成的机器码 |
| lli | IR 解释器 | 不编译，直接跑 IR |
| llvm-dis | .bc → .ll | 把二进制 IR 转成可读文本 |

3. 常用命令速查
   - clang -S -emit-llvm hello.c -o hello.ll   # C → LLVM IR
   - opt -S hello.ll -o optimized.ll            # 优化 IR
   - llc hello.ll -o hello.s                    # IR → 汇编
   - lli hello.ll                                # 直接运行 IR

4. 不需要记，需要的时候回来查
```

**验证标准**:
- [ ] 不超过 100 行
- [ ] 每个工具有一个命令示例
- [ ] 明确告诉读者"不用全记住"

---

## 文件产出汇总

| # | 文件 | 类型 | 预估行数 | 优先级 |
|---|------|------|---------|--------|
| 1 | `docs/llvm/00-环境搭建.md` | 新建 | ~80 | P0 |
| 2 | `projects/hello-pass/CMakeLists.txt` | 新建 | ~15 | P0 |
| 3 | `projects/hello-pass/HelloPass.cpp` | 新建 | ~40 | P0 |
| 4 | `projects/hello-pass/test.ll` | 新建 | ~10 | P0 |
| 5 | `projects/hello-pass/run.sh` | 新建 | ~35 | P0 |
| 6 | `projects/hello-pass/README.md` | 新建 | ~30 | P0 |
| 7 | `projects/README.md` | 修改 | +5 | P1 |
| 8 | `docs/llvm/01-LLVM-IR快速入门.md` | 新建 | ~200 | P0 |
| 9 | `docs/llvm/02-第一个LLVM-Pass.md` | 新建 | ~180 | P0 |
| 10 | `docs/llvm/03-LLVM工具箱速览.md` | 新建 | ~80 | P1 |

**总计**: 10 个文件，~675 行，全部新建（1 个修改）

---

## 验证流程

整个 Step 1+2 做完后，一个零基础读者的完整体验：

```
1. 打开 docs/llvm/00-环境搭建.md
2. 安装 LLVM → 验证 opt --version ✅
3. 进入 projects/hello-pass/
4. ./run.sh → 看到 Hello: add ✅
5. 阅读 docs/llvm/01-LLVM-IR快速入门.md
6. 用 clang -S -emit-llvm 自己生成 .ll ✅
7. 阅读 docs/llvm/02-第一个LLVM-Pass.md
8. 完成 3 个挑战 → 修改 HelloPass.cpp 并重跑 ✅
9. 阅读 docs/llvm/03-LLVM工具箱速览.md（可选）
10. 知道自己下一步该学什么 ✅
```

## 设计原则（遵循 karpathy-guidelines）

| 原则 | 在本计划中的应用 |
|------|-----------------|
| Think Before | 每个文档先确认目标读者"看完后能做什么"，再落笔 |
| Simplicity | HelloPass ≤40行，每个教程 ≤200行，不做万能框架 |
| Surgical | 只新建 docs/llvm/ 和 projects/hello-pass/，不动已有 primer 内容 |
| Goal-Driven | 每篇文档末尾有"完成后你应该能…"的验证标准 |
