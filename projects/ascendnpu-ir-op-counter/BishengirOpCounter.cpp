// ==- BishengirOpCounter.cpp - 统计 bishengir module 中 ops 类型分布 -==//
//
// 分析 Pass: 遍历 module 内所有操作，按 dialect + op 名统计计数.
//
// 对应 AscendNPU-IR:
//   模式参考: bishengir/lib/Conversion/LinalgToHFusion/LinalgToHFusion.cpp
//   同类 Pass: bishengir/lib/Conversion/ArithToHFusion/
//   测试参考: bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
//
// 如果要将此 Pass 注册到 bishengir-opt:
//   1. 复制到 bishengir/lib/Conversion/BishengirOpCounter/ 目录
//   2. 在 CMakeLists.txt 添加:
//      add_mlir_conversion_library(BishengirOpCounter BishengirOpCounter.cpp)
//   3. 在 include/bishengir/InitAllPasses.h 添加:
//      void registerBishengirOpCounterPass();
//   4. 在 tools/bishengir-opt/bishengir-opt.cpp 添加注册:
//      registerBishengirOpCounterPass();
//
// 学习目的。
// 基于 Toy Tutorial Ch3 的 ShapeInferencePass 模式 + bishengir 实际代码风格。
//   输出类似：  hfusion.elemwise_binary: 3   linalg.generic: 1   func.func: 2
//
// 注册方式：
//   在 InitAllPasses.h 中加入 registerOpCounterPass() 声明。
//   可被 bishengir-opt 用 --bishengir-op-counter 调用。
//
// 参考：
//   - Toy Tutorial Ch3: ShapeInferencePass.cpp（OperationPass 模式）
//   - bishengir InitAllPasses.h（Pass 注册方式）
//   - MLIR OpWalking 文档（walk() API）
//===

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <map>

namespace mlir::bishengir {

// ---------------------------------------------------------------------------
// OpCounterPass：统计 module 中每个 op 的出现次数
// ---------------------------------------------------------------------------
// 这是一个 OperationPass，作用于 mlir::ModuleOp（整个模块）
// 对比：
//   - Toy 的 ShapeInferencePass 作用于 func::FuncOp（单个函数）
//   - ConvertLinalgToHFusion 也是一个 OperationPass
//   - New PM 模式：继承 PassWrapper<Self, OperationPass<ModuleOp>>
// ---------------------------------------------------------------------------

class OpCounterPass
    : public PassWrapper<OpCounterPass, OperationPass<ModuleOp>> {

public:
  // Pass 的命令行名字（被 bishengir-opt --bishengir-op-counter 调用）
  StringRef getArgument() const override { return "bishengir-op-counter"; }
  StringRef getDescription() const override {
    return "统计 bishengir module 中每种 operation 的出现次数";
  }

  // -----------------------------------------------------------------------
  // runOnOperation() — MLIR Pass 的核心入口
  // 对应 Toy Tutorial Ch3: `void runOnOperation() override`
  // -----------------------------------------------------------------------
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // opCount: key = "dialect.opname", value = 计数
    std::map<std::string, int> opCount;

    // ──────── MLIR 遍历机制：walk() ────────
    //
    // walk() 是 MLIR 最常用的遍历方式：
    //   module->walk([](Operation *op) { ... });
    //
    // 它支持三种模式：
    //   1. 无 filter：遍历所有操作（嵌套 Region 内也遍历）
    //   2. 按类型 filter：walk([](arith::AddFOp op) { ... });
    //   3. walk 顺序：PreOrder（默认）或 PostOrder
    //
    // 对比 bishengir 的转换 Pass 中用的模式：
    //   转换 Pass 通常用 PatternRewriter 的 matchAndRewrite，
    //   而分析 Pass 用 walk() 就够了。
    // ─────────────────────────────────────

    module->walk([&](Operation *op) {
      // 每个 MLIR Operation 都有：
      //   1. getName() — 返回 OperationName（如 "hfusion.elemwise_binary"）
      //   2. getDialect() — 返回所属 Dialect 的 Namespace
      //   3. getNumOperands() / getNumResults() — 操作数/结果数

      std::string opName = op->getName().getStringRef().str();
      opCount[opName]++;

      // 也可以按 dialect namespace 分组：
      //   if (auto *dialect = op->getDialect())
      //     dialectCount[dialect->getNamespace().str()]++;
    });

    // ──────── 输出统计结果 ────────

    llvm::outs() << "━━━ Bishengir OpCounter ━━━\n";
    llvm::outs() << "  Total ops: " << opCount.size() << " unique types\n\n";

    int total = 0;
    for (const auto &[name, count] : opCount) {
      llvm::outs().indent(2) << name << ": " << count << "\n";
      total += count;
    }

    llvm::outs() << "\n  ─────────────────────\n";
    llvm::outs() << "  Grand total: " << total << " operations\n";
    llvm::outs() << "━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // 标记分析成功（对比转换 Pass 的 signalPassFailure）
    // 分析 Pass 不会修改 IR，所以总是 success
  }
};

// ---------------------------------------------------------------------------
// Pass 注册函数（在 InitAllPasses.h 中添加声明）
// ---------------------------------------------------------------------------
//
// 在 InitAllPasses.h 中添加：
//   void registerOpCounterPass();
//
// 调用方式（在 bishengir-opt 中）：
//   registry.registerOpCounterPass();
//   // 或直接在 bishengir-opt main 中：
//   PassRegistration<OpCounterPass>("bishengir-op-counter", "...");
//
// ---------------------------------------------------------------------------
void registerOpCounterPass() {
  PassRegistration<OpCounterPass> reg;
}

} // namespace mlir::bishengir
