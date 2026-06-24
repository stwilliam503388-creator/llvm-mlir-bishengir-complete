# 04 — 写一个会修改 IR 的 Pass

> 目标：从"只读不改"的 HelloPass 升级到"会改 IR"的 OptPass
> 前置：[02 — 第一个 LLVM Pass](./02-第一个LLVM-Pass.md)
> 预估时间：15 分钟

## 1. 从 HelloPass 到 OptPass

HelloPass 的核心：`return false` — 告诉 LLVM "我没改 IR"。

现在要做的就是：**改 IR，然后 `return true`**。

| | HelloPass | 本次要写的 OptPass |
|---|---|---|
| 行为 | 打印函数信息 | 删除死代码 |
| 返回值 | `return false` | `return true` |
| LLVM 反应 | 不重新分析 | 重新分析（因为 IR 变了） |
| 验证 | 看终端输出 | 对比 `.ll` 文件优化前后 |

## 2. 操作 IR 的三个基本动作

### 删除指令

```cpp
// I 是一条指令
I.eraseFromParent();  // 从父 BasicBlock 中移除
```

### 替换指令

```cpp
// 把指令 I 的所有使用者替换为 V
I.replaceAllUsesWith(V);
I.eraseFromParent();
```

### 遍历 + 安全删除

```cpp
// ❌ 错误：删除时迭代器失效
for (auto &I : BB) {
    if (isDead(I)) I.eraseFromParent();  // 崩！
}

// ✅ 正确：先收集，再删除
SmallVector<Instruction *, 8> toDelete;
for (auto &I : BB)
    if (isDead(I)) toDelete.push_back(&I);
for (auto *I : toDelete)
    I.eraseFromParent();
```

**核心规则**：不要在遍历容器时删除元素。先收集到列表，再统一删除。

## 3. 完整示例：删除无用 alloca

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
struct DCEPass : public PassInfoMixin<DCEPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool changed = false;

    for (auto &BB : F) {
      SmallVector<Instruction *, 8> toDelete;
      for (auto &I : BB) {
        // alloca 指令：分配栈空间
        // 如果没有任何人使用它 → 死代码
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          if (AI->use_empty()) {
            toDelete.push_back(AI);
            changed = true;
          }
        }
      }
      for (auto *I : toDelete)
        I.eraseFromParent();
    }

    return changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
  }
};
}
```

**逐行解读**：

| 行 | 在干什么 |
|----|---------|
| `dyn_cast<AllocaInst>(&I)` | 判断这条指令是不是 `alloca` |
| `AI->use_empty()` | 有没有人用这个 alloca？ |
| `toDelete.push_back(AI)` | 先收集（不立即删） |
| `I.eraseFromParent()` | 统一删除 |
| `PreservedAnalyses::none()` | IR 变了，所有分析失效 |
| `PreservedAnalyses::all()` | IR 没变，分析可复用 |

## 4. 和 HelloPass 的精确对照

| | HelloPass | 上面的 DCEPass |
|---|---|---|
| 遍历 | `for (auto &BB : F)` | 相同 |
| 判断 | `F.getName()` | `dyn_cast<AllocaInst>` |
| 修改 | 不改 | `eraseFromParent()` |
| 返回值 | `PreservedAnalyses::all()` | `none()` 或 `all()` |
| 多趟优化 | 不需要 | LLVM 会重复跑直到 IR 稳定 |

> `return PreservedAnalyses::none()` 告诉 LLVM "IR 变了"，LLVM 会重新分析依赖关系并可能再跑一次这个 Pass。

## 验证

- [ ] 能说出 `return false` vs `return true` 的区别
- [ ] 知道删除指令的安全模式（先收集再删）
- [ ] 能写出删除无用指令的代码

> 📖 [术语表](../glossary.md)
> **下一步**：→ [opt-pass 动手项目](../../projects/opt-pass/) — 写一个有实际效果的 Pass
