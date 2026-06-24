# 05 — 常见 Pass 模式

> 目标：掌握三种最常见的 Pass 模式，能自己写简单优化
> 前置：[04 — 写一个会修改 IR 的 Pass](./04-写一个会修改IR的Pass.md)
> 预估时间：20 分钟

## 1. 模式一：死代码消除（DCE）

**目标**：删除定义后从未被使用的指令。

```llvm
; 输入
define i32 @dce_example() {
  %x = add i32 1, 2    ; x 定义后从未使用 → 死代码
  ret i32 42
}

; 输出
define i32 @dce_example() {
  ret i32 42           ; x 被删除
}
```

**代码骨架**：

```cpp
for (auto &I : BB) {
  if (I.use_empty() && !I.mayHaveSideEffects())
    toDelete.push_back(&I);
}
```

| 条件 | 含义 |
|------|------|
| `use_empty()` | 没人用这条指令的结果 |
| `!mayHaveSideEffects()` | 没有副作用（不是 store/call/br） |

## 2. 模式二：常量折叠

**目标**：编译期算出常量表达式的结果。

```llvm
; 输入
define i32 @fold_example() {
  %x = add i32 3, 5    ; 编译期就能算出来 = 8
  ret i32 %x
}

; 输出
define i32 @fold_example() {
  ret i32 8             ; 直接替换为结果
}
```

**代码骨架**：

```cpp
if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
  if (auto *C1 = dyn_cast<ConstantInt>(BO->getOperand(0)))
    if (auto *C2 = dyn_cast<ConstantInt>(BO->getOperand(1))) {
      // 两个操作数都是常量 → 编译期算出结果
      Constant *result = ConstantExpr::get(BO->getOpcode(), C1, C2);
      I.replaceAllUsesWith(result);
      toDelete.push_back(&I);
    }
}
```

**厨房类比**：菜谱上写"3 个鸡蛋 + 5 个鸡蛋"。编译器在备菜阶段就算出"8 个鸡蛋"，不用等到炒菜时再数。

## 3. 模式三：指令合并

**目标**：消除无意义的运算。

```llvm
; 输入
define i32 @combine_example(i32 %a) {
  %x = add i32 %a, 0    ; +0 无意义
  %y = mul i32 %x, 1    ; ×1 无意义
  ret i32 %y
}

; 输出
define i32 @combine_example(i32 %a) {
  ret i32 %a              ; a+0*1 = a
}
```

**代码骨架**：

```cpp
if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
  // add X, 0 → X
  if (BO->getOpcode() == Instruction::Add) {
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)))
      if (C->isZero()) {
        I.replaceAllUsesWith(BO->getOperand(0));
        toDelete.push_back(&I);
      }
  }
}
```

## 4. 三种模式对比

| | DCE | 常量折叠 | 指令合并 |
|---|---|---|---|
| 做什么 | 删掉不用的 | 编译期算出来 | 消掉冗余运算 |
| 判断条件 | `use_empty()` | 两个操作数都是常量 | 操作数是 0/1 |
| 怎么改 | `eraseFromParent()` | `replaceAllUsesWith` + 删除 | 同样 |
| 真实 LLVM 里 | `-dce` pass | `-constprop` pass | `-instcombine` pass |

**这三个模式可以组合**：常量折叠后可能产生新的死代码 → DCE 再跑一次 → 更干净。

## 验证

- [ ] 能写出 DCE 的判断条件（`use_empty()`）
- [ ] 能写出常量折叠的判断条件（两个操作数都是 `ConstantInt`）
- [ ] 能说出"先收集再删除"的原因

> 📖 [术语表](../glossary.md)
> **下一步**：[06 — Pass 调试与测试](./06-Pass调试与测试.md)
