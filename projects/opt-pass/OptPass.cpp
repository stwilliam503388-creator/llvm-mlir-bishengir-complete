#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace {
struct OptPass : public PassInfoMixin<OptPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool changed = false;
    for (auto &BB : F) {
      SmallVector<Instruction *, 8> toDelete;
      for (auto &I : BB) {
        if (I.use_empty() && !I.mayHaveSideEffects()
            && !I.isTerminator()) {
          toDelete.push_back(&I);
          changed = true;
        }
      }
      for (auto *I : toDelete) {
        errs() << "  DCE: removing " << *I << "\n";
        I->eraseFromParent();
      }
    }
    return changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
  }
};
}

extern "C" {
  struct PassPluginLibraryInfo {
    uint32_t APIVersion;
    const char *PluginName;
    const char *PluginVersion;
    void (*RegisterPassBuilderCallbacks)(PassBuilder &);
  };
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {2, "OptPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
              [](StringRef Name, FunctionPassManager &FPM,
                 ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "opt-pass") {
                  FPM.addPass(OptPass());
                  return true;
                }
                return false;
              });
          }};
}
