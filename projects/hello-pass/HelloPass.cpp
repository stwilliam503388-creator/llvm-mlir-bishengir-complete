#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  struct HelloPass : public PassInfoMixin<HelloPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
      errs() << "Hello: " << F.getName() << "\n";
      errs() << "  参数数量: " << F.arg_size() << "\n";
      errs() << "  基本块数量: " << F.size() << "\n";
      return PreservedAnalyses::all();
    }
  };
}

// 手动声明 PassPlugin 接口（Homebrew LLVM 22 未安装 PassPlugin.h）
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
  return {
    2, "HelloPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "hello") {
            FPM.addPass(HelloPass());
            return true;
          }
          return false;
        });
    }
  };
}
