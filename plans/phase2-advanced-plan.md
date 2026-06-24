# Phase 2 进阶 — 写有实际效果的 LLVM Pass

> Status: planning | Created: 2026-06-21

## 定位

Phase 2 基础教的是"看懂 IR + 写 HelloWorld Pass（只读不改）"。
Phase 2 进阶教的是"写会修改 IR 的 Pass，做出实际优化效果"。

## 和 Phase 2 基础的对照

| | Phase 2 基础 | Phase 2 进阶 |
|---|---|---|
| Pass 行为 | 只打印（`return false`） | 修改 IR（`return true`） |
| 做什么 | 统计函数/基本块信息 | 删除死代码、折叠常量 |
| 动手项目 | hello-pass | opt-pass（新建） |
| 验证方式 | 看终端输出 | 对比 .ll 文件优化前后 |

## 内容设计（3 篇文档 + 1 个项目）

### 文档 1：`docs/llvm/04-写一个会修改IR的Pass.md`

| 章节 | 内容 |
|------|------|
| 从 HelloPass 到 OptPass | `return false` vs `return true` 的区别 |
| 怎么删除一条指令 | `I.eraseFromParent()` |
| 怎么替换一条指令 | `ReplaceInstWithValue()` |
| 怎么遍历 + 修改 | `for (auto &I : instructions(BB))`，注意迭代器失效 |
| 完整示例：删除 `ret void` 前无用的 `alloca` | 10 行代码 |

### 文档 2：`docs/llvm/05-常见Pass模式.md`

| 章节 | 内容 |
|------|------|
| 死代码消除 | 找到没有被使用的指令 → 删除 |
| 常量折叠 | `add i32 3, 5` → `8` |
| 指令合并 | `%x = add %a, 0` → `%x = %a` |
| 每个模式配代码片段 + 输入 .ll + 预期输出 .ll |

### 文档 3：`docs/llvm/06-Pass调试与测试.md`

| 章节 | 内容 |
|------|------|
| 用 `opt -S` 对比优化前后 IR | 最直观的验证 |
| 写测试用例：`; RUN: opt -passes="mypass" %s \| FileCheck %s` | LLVM 标准测试 |
| 常见错误：迭代器失效、use-list 修改 | 每个配修复方法 |

### 项目：`projects/opt-pass/`

对标 hello-pass 的结构，但实现一个**死代码消除 Pass**：

| 文件 | 说明 |
|------|------|
| `OptPass.cpp` | 40 行：遍历指令 → 检查是否有使用者 → 无则删除 |
| `test.ll` | 含死代码的 IR（定义但未使用的变量） |
| `test_expected.ll` | 删除死代码后的 IR |
| `run.sh` | 一键构建 + `opt -S` 对比输出 |
| `README.md` | 说明 + 预期输出对照 |

### 学习路径

```
Phase 2 基础结束 → llvm/03-LLVM工具箱速览
       ↓
llvm/04-写一个会修改IR的Pass (15 min)
       ↓
projects/opt-pass/ (动手, 30 min)
       ↓
llvm/05-常见Pass模式 (20 min)
       ↓
llvm/06-Pass调试与测试 (15 min)
```

### 总工作量

| 类别 | 文件数 | 新写行数 |
|------|--------|---------|
| 文档 | 3 篇 | ~600 |
| 项目 | opt-pass（5 文件） | ~120 |
| 全局更新 | llvm/03 末尾 + quickstart | +5 |

**总计**：8 个文件操作，~725 行，预估 1-2 小时。
