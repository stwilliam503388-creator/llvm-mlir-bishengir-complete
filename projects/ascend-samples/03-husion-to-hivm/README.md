# 用例 3 — husion-to-hivm

husion.elemwise_binary "add" → hivm.load + hivm.vadd + hivm.store。

husion 是融合后的高级 IR，hivm 是接近硬件的虚拟指令。

## 关键变化

| husion | hivm |
|--------|------|
| 隐式数据搬运（tensor） | 显式搬运（memref + load/store） |
| 一个操作 `elemwise_binary` | 多个操作 `load → vadd → store` |
| 适合融合优化 | 适合代码生成 |

## 学完后

→ 用例 4：hivm → LLVM IR
