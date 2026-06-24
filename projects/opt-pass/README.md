# OptPass — 死代码消除 Pass

从 HelloPass 升级：不再只是打印，而是真正修改 IR。

## 快速开始

```bash
chmod +x run.sh
./run.sh
```

## 预期输出

```
DCE: removing   %x = add i32 1, 2
DCE: removing   %y = alloca i32
```

优化前后对比：

| 函数 | 优化前 | 优化后 |
|------|--------|--------|
| `has_dead_code` | 3 条指令（含 2 条死代码） | 1 条 `ret i32 42` |
| `all_used` | 2 条指令（都活着） | 不变 |

## 和 hello-pass 的对照

| | hello-pass | opt-pass |
|---|---|---|
| 做什么 | 打印函数信息 | 删除死代码 |
| 返回值 | `PreservedAnalyses::all()` | `none()`（IR 变了） |
| 遍历 | `for (BB : F)` | 相同 |
| 修改 IR | 不改 | `eraseFromParent()` |

## 文件说明

| 文件 | 作用 |
|------|------|
| OptPass.cpp | 死代码消除实现（35 行） |
| test.ll | 测试输入（含死代码） |
| test_expected.ll | 预期输出（死代码已删除） |
| run.sh | 一键构建 + 对比 |
