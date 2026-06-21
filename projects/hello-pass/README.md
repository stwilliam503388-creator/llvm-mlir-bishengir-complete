# HelloPass — 你的第一个 LLVM Pass

最简单的 LLVM FunctionPass：遍历每个函数，打印函数名、参数数量和基本块数量。

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
| test.ll | 测试输入的 LLVM IR |
| run.sh | 一键构建 + 运行 |

## 学完这个，下一步？

- 深入理解 Pass 机制 → [第一个 LLVM Pass（教学版）](../../docs/llvm/02-第一个LLVM-Pass.md)
- 遇到不认识的术语 → [术语表](../../docs/glossary.md)
- 想知道为什么学这些 → [为什么学 Ascend NPU 编译器？](../../docs/why-ascend.md)
