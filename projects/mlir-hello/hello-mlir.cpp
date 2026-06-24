#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;

struct HelloMLIRPass : public PassWrapper<HelloMLIRPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    int opCount = 0;
    func.walk([&](Operation *) { opCount++; });
    std::string msg = "Hello: " + func.getName().str() + "\n"
                    + "  Operation 数量: " + std::to_string(opCount) + "\n";
    llvm::errs() << msg;
  }
};

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect>();
  MLIRContext ctx(registry);

  if (argc < 2) {
    llvm::errs() << "Usage: hello-mlir <input.mlir>\n";
    return 1;
  }

  auto src = openInputFile(argv[1]);
  if (!src) { llvm::errs() << "Cannot open " << argv[1] << "\n"; return 1; }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(src), SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) return 1;

  PassManager pm(&ctx);
  pm.addNestedPass<func::FuncOp>(std::make_unique<HelloMLIRPass>());

  LogicalResult result = pm.run(*module);
  return failed(result);
}
