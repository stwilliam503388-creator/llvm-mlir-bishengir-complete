// ==- standalone-opt.cpp - 最简编译版本 -==//
// 使用 MLIR 的 Op 模板，不定义 fold/parse/print。

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

// ============================================================
// 最简单 Op 定义 — 只声明操作名，没有自定义逻辑
// ============================================================
namespace mlir::standalone {

using namespace mlir;

class ConstantOp : public Op<ConstantOp> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.constant"; }
};

class AddOp : public Op<AddOp, OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.add"; }
};

class MulOp : public Op<MulOp, OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.mul"; }
};

class TransposeOp : public Op<TransposeOp, OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.transpose"; }
};

class PrintOp : public Op<PrintOp> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.print"; }
};

class ReturnOp : public Op<ReturnOp> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.return"; }
};

// ============================================================
// Dialect — 注册所有 Op
// ============================================================
class StandaloneDialect : public Dialect {
public:
  explicit StandaloneDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<StandaloneDialect>()) {
    addOperations<ConstantOp, AddOp, MulOp, TransposeOp, PrintOp, ReturnOp>();
  }
  static StringRef getDialectNamespace() { return "standalone"; }
};

} // namespace mlir::standalone

// ============================================================
// OpCounter Pass
// ============================================================
class StandaloneOpCounterPass
    : public mlir::PassWrapper<StandaloneOpCounterPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const override { return "count-ops"; }
  StringRef getDescription() const override { return "count ops"; }
  void runOnOperation() override {
    auto module = getOperation();
    std::map<std::string, int> counts;
    module->walk([&](mlir::Operation *op) {
      counts[op->getName().getStringRef().str()]++;
    });
    llvm::outs() << "━━━ OpCounter ━━━\n";
    int total = 0;
    for (auto &[name, cnt] : counts) {
      llvm::outs().indent(2) << name << ": " << cnt << "\n";
      total += cnt;
    }
    llvm::outs() << "  Total: " << total << " ops\n";
  }
};

// ============================================================
// TransposeElim Pass
// ============================================================
struct TransposeTransposeElim
    : public mlir::OpRewritePattern<mlir::standalone::TransposeOp> {
  TransposeTransposeElim(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::standalone::TransposeOp>(context) {}
  mlir::LogicalResult matchAndRewrite(
      mlir::standalone::TransposeOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto parent = op.getInput()
                      .template getDefiningOp<mlir::standalone::TransposeOp>();
    if (!parent) return mlir::failure();
    rewriter.replaceOp(op, parent.getInput());
    return mlir::success();
  }
};

class StandaloneTransposeElimPass
    : public mlir::PassWrapper<StandaloneTransposeElimPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  StringRef getArgument() const override { return "elim-transpose"; }
  StringRef getDescription() const override { return "elim transpose"; }
  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TransposeTransposeElim>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

// ============================================================
// main
// ============================================================
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::PassRegistration<StandaloneOpCounterPass>();
  mlir::PassRegistration<StandaloneTransposeElimPass>();

  mlir::DialectRegistry registry;
  registry.insert<mlir::standalone::StandaloneDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone MLIR optimizer", registry));
}
