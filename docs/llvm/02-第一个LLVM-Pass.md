# 02 - 第一个 LLVM Pass

> 目标：理解 Pass 的运作方式，能自己修改 HelloPass 并重跑
> 前置：[01 - LLVM IR 快速入门](./01-LLVM-IR快速入门.md)、[HelloPass 项目](../../projects/hello-pass/)
> 预估时间：40 分钟

---

## 1. 回顾：Pass 在编译流水线中的角色

在 [Primer 02 - Pass 与 Lowering](../primer/02-Pass与Lowering.md) 里我们比喻 Pass 是"质检员"——

每个 Pass 对 IR 做一件事：检查、优化、或者变换。多个 Pass 串在一起就是完整的编译流水线。

LLVM 的 Pass 就是这种"质检员"的具体实现。

---

## 2. 从 HelloPass 出发：逐行解读

先确保你能跑通 HelloPass：

```bash
cd projects/hello-pass
chmod +x run.sh
./run.sh
```

预期输出：
```
Hello: add
  参数数量: 2
  基本块数量: 1
Hello: say_hello
  参数数量: 0
  基本块数量: 1
```

现在逐行解读 `HelloPass.cpp`：

```cpp
#include "llvm/Pass.h"           // Pass 基类
#include "llvm/IR/Function.h"    // Function 类型
#include "llvm/Support/raw_ostream.h"  // errs() — LLVM 的"打印到屏幕"
```

| include | 作用 | 没有它会怎样 |
|---------|------|-------------|
| Pass.h | 提供 FunctionPass 基类 | 编译器不认识 FunctionPass |
| Function.h | 让你能访问函数名、参数、基本块 | F.getName() 报错 |
| raw_ostream.h | 提供 errs() 输出流 | errs() 报错 |

```cpp
struct HelloPass : public FunctionPass {
    // ↑ "我是一个 FunctionPass"
    //    意味着 LLVM 会给你的 runOnFunction() 喂每个函数
```

```cpp
    bool runOnFunction(Function &F) override {
        errs() << "Hello: " << F.getName() << "\n";
        //            ↑ 函数名     ↑ LLVM 的 "打印到终端"
        //              等价于 C++ 的 std::cout
```

```cpp
        errs() << "  参数数量: " << F.arg_size() << "\n";
        errs() << "  基本块数量: " << F.size() << "\n";
        //     ↑ 参数个数            ↑ 基本块个数
```

```cpp
        return false;
        // ↑ "我没有修改 IR"
        //    如果改了 IR，应该 return true，告诉 LLVM 重新分析
    }
```

```cpp
static RegisterPass<HelloPass> X("hello", "Hello World Pass",
                                  false, false);
//  ↑ 注册成名为 "hello" 的 Pass
//    运行时用: opt --passes="hello"
```

---

## 3. 动手挑战

### 挑战 1：打印每个基本块的指令数量

在 `runOnFunction()` 中添加：

```cpp
for (auto &BB : F) {
  errs() << "  基本块 " << BB.getName()
         << " 有 " << BB.size() << " 条指令\n";
}
```

**预期新增输出**：
```
Hello: add
  参数数量: 2
  基本块数量: 1
  基本块  有 2 条指令         ← 新增（add + ret = 2 条）
Hello: say_hello
  参数数量: 0
  基本块数量: 1
  基本块  有 1 条指令         ← 新增（只有 ret）
```

### 挑战 2：统计函数中有几个 add 指令

```cpp
int addCount = 0;
for (auto &BB : F) {
  for (auto &I : BB) {
    if (StringRef(I.getOpcodeName()) == "add") {
      addCount++;
    }
  }
}
errs() << "  add 指令数量: " << addCount << "\n";
```

**预期新增输出**：
```
Hello: add
  ...
  add 指令数量: 1              ← 只有一个 add 指令
Hello: say_hello
  ...
  add 指令数量: 0              ← say_hello 里没有 add
```

### 挑战 3：找参数最多的函数

```bash
# 用 grep 从 test.ll 中找所有 define 行
grep "^define" test.ll
# 应输出:
# define i32 @add(i32 %a, i32 %b)
# define void @say_hello()
```

当前 `add` 有 2 个参数，`say_hello` 有 0 个。

**试试写一个 5 参数的函数**，加到 `test.ll` 末尾：

```llvm
define i32 @sum5(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  %abcd = add i32 %abc, %d
  %abcde = add i32 %abcd, %e
  ret i32 %abcde
}
```

重新运行 `./run.sh`，你应该看到：

```
Hello: sum5
  参数数量: 5                  ← 最高！
  基本块数量: 1
  add 指令数量: 4              ← 如果完成了挑战2
```

---

## 4. Pass 的类型简介

| 类型 | 粒度 | 回调函数 | 什么时候用 |
|------|------|---------|-----------|
| FunctionPass | 每个函数 | runOnFunction() | 分析/优化一个函数内部 |
| ModulePass | 整个模块 | runOnModule() | 跨函数分析、添加/删除函数 |
| BasicBlockPass | 每个基本块 | runOnBasicBlock() | 局部优化 |

**不需要现在就记住**。先写 FunctionPass，遇到需要跨函数操作时再查 ModulePass。

---

## 5. 如果改错了怎么办？

如果改了 HelloPass.cpp 编译报错：

1. **先看错误的第一行** — 编译器的第一行报错通常是根因
2. **忘记 `#include`** — 最常见的错误，用到了 Function 但忘了 include Function.h
3. **变量名写错** — LLVM 变量以 `%` 开头，C++ 变量以字母开头，注意区分

---

## 下一步

→ [03 - LLVM 工具箱速览](./03-LLVM工具箱速览.md) — 了解 LLVM 全家桶，知道每个工具干什么

---

## 本节验证清单

- [ ] 能跑通 HelloPass（`./run.sh` 看到预期输出）
- [ ] 完成了挑战 1（打印基本块指令数）
- [ ] 完成了挑战 2（统计 add 指令数）
- [ ] 完成了挑战 3（写了一个多参数函数测试）
- [ ] 能解释 `return false` 的含义（没有修改 IR）

> 📖 遇到不认识的术语？→ [术语表](../glossary.md)
