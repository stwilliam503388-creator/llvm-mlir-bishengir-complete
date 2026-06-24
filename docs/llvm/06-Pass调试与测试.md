# 06 — Pass 调试与测试

> 目标：能验证自己写的 Pass 是否正确
> 前置：[05 — 常见 Pass 模式](./05-常见Pass模式.md)
> 预估时间：15 分钟

## 1. 最直观的验证：对比 .ll 文件

```bash
# 用 opt 跑你的 Pass，输出优化后的 IR
opt -load-pass-plugin ./libMyPass.dylib \
    --passes="mypass" \
    input.ll -S -o output.ll

# 对比
diff input.ll output.ll
```

这就是 opt-pass 项目用的方式：`run.sh` 里跑 `opt -S`，对比输出和预期。

## 2. LLVM 标准测试：FileCheck

```llvm
; test_dce.ll — LLVM 标准测试格式
; RUN: opt -load-pass-plugin ./libDCEPass.dylib --passes="dce" %s -S | FileCheck %s

define i32 @test() {
; CHECK-NOT: add
; CHECK: ret i32 42
  %x = add i32 1, 2    ; 应被删除
  ret i32 42
}
```

| 指令 | 含义 |
|------|------|
| `; RUN:` | 告诉测试框架怎么跑这个测试 |
| `; CHECK:` | 输出中必须包含这行 |
| `; CHECK-NOT:` | 输出中不能包含这行 |
| `%s` | 替换为当前文件名 |

```bash
# 运行单个测试
opt -load-pass-plugin ./libDCEPass.dylib \
    --passes="dce" test_dce.ll -S | FileCheck test_dce.ll
```

## 3. 常见错误

| 错误 | 现象 | 修复 |
|------|------|------|
| 迭代器失效 | `Segfault` 或 `Assertion failed` | 先收集到 `SmallVector`，再统一删除 |
| 删了还有使用者 | `Assertion failed: use_empty()` | 先 `replaceAllUsesWith`，再 `eraseFromParent` |
| 忘了 return true | 优化只跑一次就停了 | `return PreservedAnalyses::none()` 触发重跑 |
| use-list 顺序 | 结果每次不一样 | 用稳定容器（`SmallVector` 而非 `SmallPtrSet`） |

## 4. 调试技巧

### 打印 IR 变化

```cpp
// 删除前后各打印一次 BB
errs() << "Before:\n" << BB << "\n";
// ... 删除 ...
errs() << "After:\n" << BB << "\n";
```

### 只跑一个 Pass

```bash
# --passes="mypass" 只跑你的 Pass，不跑其他优化
opt -load-pass-plugin ./libMyPass.dylib --passes="mypass" input.ll -S
```

### 看 Pass 跑了几次

```bash
# --debug-pass-manager 看调度信息
opt -load-pass-plugin ./libMyPass.dylib --passes="mypass" input.ll -S \
    --debug-pass-manager 2>&1 | grep mypass
```

## 验证

- [ ] 能用 `opt -S` 对比优化前后的 IR
- [ ] 能写一个简单的 FileCheck 测试
- [ ] 知道"先收集再删除"的原因

> 📖 [术语表](../glossary.md)
> **下一步**：[Phase 3 — MLIR 学习](../mlir/README.md)
