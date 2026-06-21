# LLVM Pass 开发

> 对 Ascend 后端开发：自定义 LLVM Pass 是实现 NPU 特定代码生成和优化的核心手段。

## Pass 的两种类型

| 类型 | 行为 | PreservedAnalyses | 示例 |
|------|------|-------------------|------|
| 分析 Pass | 只读 IR | 返回 `all()` | 统计指令数 |
| 变换 Pass | 修改 IR | 返回 `none()` | 常量折叠 |

## Pass 骨架（New PM）

```cpp
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct MyPass : PassInfoMixin<MyPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    errs() << "Function: " << F.getName() << "\\n";
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "my-pass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM, auto) {
                  if (Name == "my-pass") { FPM.addPass(MyPass()); return true; }
                  return false;
                });
          }};
}
```

## 编译

```bash
export LLVM_DIR="/opt/homebrew/opt/llvm"
$LLVM_DIR/bin/clang++ -std=c++17 -fno-rtti -shared \\
  -I$LLVM_DIR/include -fPIC \\
  MyPass.cpp -L$LLVM_DIR/lib -Wl,-undefined,dynamic_lookup \\
  -o libMyPass.dylib
```

## 运行

```bash
opt -load-pass-plugin libMyPass.dylib -passes="my-pass" -disable-output input.ll
```

## BBCounter 实战

```cpp
struct BBCounter : PassInfoMixin<BBCounter> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    int count = 0;
    for (auto &BB : F) {   // 遍历基本块
      (void)BB; count++;
    }
    errs() << "[BBCounter] " << F.getName()
           << " | basic blocks: " << count << "\\n";
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};
```

## 关键 API

| API | 用途 |
|-----|------|
| `Function::getName()` | 函数名 |
| `for (auto &BB : Func)` | 遍历基本块 |
| `for (auto &I : BB)` | 遍历指令 |
| `Instruction::getOpcode()` | 指令操作码 |
| `PreservedAnalyses::all()` | "没改" |
| `PreservedAnalyses::none()` | "改了" |
