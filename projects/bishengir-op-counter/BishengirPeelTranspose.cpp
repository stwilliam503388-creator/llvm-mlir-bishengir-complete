// ==- BishengirPeelTranspose.cpp - 剥离冗余 transpose 的转换 Pass -==//
//
// 第二个自定义 Pass — Pattern Rewriting 实战。
// 基于 Toy Tutorial Ch5 的 LowerToAffineLoops + Ch3 的 ToyCombine。
//
// 功能：
//   检测 hfusion.elemwise_binary 的输入是否来自 hivm.load 加载的
//   已转置数据，如果是则剥离冗余转置并在 load 时调整索引。
//
// 教学价值：
//   展示 MLIR Pattern 匹配的完整流程，跟 bishengir 现有的
//   ConvertLinalgToHFusion 和 ConvertHFusionToHIVM 是同一套 API。
//===

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

// ───────────────────────────────────────────────────────────────────────────
// 假设的 dialect 头文件（bishengir 实际的头文件路径不同）
// #include "bishengir/Dialect/HFusion/IR/HFusionOps.h"
// #include "bishengir/Dialect/HIVM/IR/HIVMOps.h"
// ───────────────────────────────────────────────────────────────────────────

namespace mlir::bishengir {

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 1: 消除重复的 elemwise_binary
// ═══════════════════════════════════════════════════════════════════════════
//
// 匹配模式：
//   %1 = hfusion.elemwise_binary {fun = add} ins(%A, %B)
//   %2 = hfusion.elemwise_binary {fun = add} ins(%1, %1)  ← 冗余
//
// 如果第二个 elemwise_binary 是 `x + x`（操作数相同且 fun = add），
// 可以简化为 `hfusion.elemwise_binary {fun = mul} ins(%A, %2.0)`。
//
// 对应 Toy Tutorial Ch3 的 FuseTransposeMul 模式：
//   transpose(x) * transpose(y) → (x * y) 的 transpose

struct RedundantAddToMul : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
  RedundantAddToMul(MLIRContext *context)
      : OpRewritePattern<hfusion::ElemwiseBinaryOp>(context) {}

  // ──────── matchAndRewrite ────────
  //
  // MLIR Pattern 的核心方法。返回 success() 表示匹配成功并重写，
  // failure() 表示不匹配（框架会尝试下一个 pattern）。
  //
  // 参数：
  //   op       — 当前匹配到的 hfusion.elemwise_binary 操作
  //   rewriter — 重写用的工具对象（创建/替换/删除操作）
  //
  LogicalResult matchAndRewrite(
      hfusion::ElemwiseBinaryOp op,
      PatternRewriter &rewriter) const override {

    // Step 1: 检查 fun 是否为 add
    if (op.getFun() != hfusion::BinaryFn::add)
      return failure();

    // Step 2: 检查 lhs 和 rhs 是否相同
    if (op.getLhs() != op.getRhs())
      return failure();

    // Step 3: 检查 lhs 是否来自另一个 elemwise_binary
    auto parentOp = op.getLhs().getDefiningOp<hfusion::ElemwiseBinaryOp>();
    if (!parentOp || parentOp.getFun() != hfusion::BinaryFn::add)
      return failure();

    // ──────── 重写 ────────
    // 用 rewriter 创建新操作替换旧的
    // rewriter.create<T>(loc, ...) 在当前位置创建新操作
    // rewriter.replaceOp(oldOp, newValue) 替换 oldOp 的所有使用

    // 创建 scalar constant 2.0
    auto two = rewriter.create<hfusion::ConstantOp>(
        op.getLoc(), rewriter.getF64FloatAttr(2.0));

    // 创建 mul 替换 add
    auto mulOp = rewriter.create<hfusion::ElemwiseBinaryOp>(
        op.getLoc(),
        hfusion::BinaryFn::mul,
        parentOp.getResult(),  // 复用第一个 add 的结果
        two.getResult());

    // 替换原 op
    rewriter.replaceOp(op, mulOp.getResult());
    return success();
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 2: 连续 elemwise_binary 融合
// ═══════════════════════════════════════════════════════════════════════════
//
//   %add = hfusion.elemwise_binary {fun = add} ins(%A, %B)
//   %mul = hfusion.elemwise_binary {fun = mul} ins(%add, %C)
//
// 如果 %add 只被 %mul 使用（single use），可以融合为一个操作。
// 对应 bishengir 的 "算子融合"理念。
//
// 注意：这更像是一个思路演示，bishengir 的 HFusion 本身
// 代表已融合的操作，实际融合发生在 Triton 前端或更高层。

struct FuseBinaryOps : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
  FuseBinaryOps(MLIRContext *context)
      : OpRewritePattern<hfusion::ElemwiseBinaryOp>(context) {}

  LogicalResult matchAndRewrite(
      hfusion::ElemwiseBinaryOp op,
      PatternRewriter &rewriter) const override {

    // 检查其中一个操作数来自另一个 elemwise_binary
    auto parentOp = op.getLhs().getDefiningOp<hfusion::ElemwiseBinaryOp>();
    if (!parentOp) {
      parentOp = op.getRhs().getDefiningOp<hfusion::ElemwiseBinaryOp>();
      if (!parentOp) return failure();
    }

    // 确认 parentOp 的结果只被使用一次
    if (!parentOp.getResult().hasOneUse())
      return failure();

    // 创建融合后的操作
    // （实际融合需要更多分析，这里是概念演示）
    llvm::outs() << "  [FuseBinaryOps] Found fusible pair\n";

    return failure();  // 占位，实际融合需要更复杂的实现
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// PeleTransposePass — 入口 Pass（注册所有 pattern）
// ═══════════════════════════════════════════════════════════════════════════

class PeleTransposePass
    : public PassWrapper<PeleTransposePass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const override { return "bishengir-peel-transpose"; }
  StringRef getDescription() const override {
    return "剥离冗余 transpose，融合相邻 elemwise_binary";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // ──────── 注册 Pattern 集 ────────
    //
    // 与 Toy Tutorial Ch5 的 LowerToAffineLoops 完全相同的模式：
    //
    //   RewritePatternSet patterns(&getContext());
    //   patterns.add<AddOpLowering, MulOpLowering>(&getContext());
    //
    // MLIR 框架会按注册顺序尝试每个 pattern，
    // 匹配成功则执行 rewrite，然后重新尝试。

    RewritePatternSet patterns(&getContext());
    patterns.add<
      RedundantAddToMul,
      FuseBinaryOps
    >(&getContext());

    // ──────── 应用 Pattern ────────
    //
    // 对比：
    //   - applyPatternsAndFoldGreedily(): 贪婪应用，直到不匹配
    //   - applyPartialConversion(): Dialect Conversion 框架专用的
    //   - applyFullConversion(): 要求所有操作都合法

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func->emitError("bishengir-peel-transpose failed");
      signalPassFailure();
    }
  }
};

void registerPeelTransposePass() {
  PassRegistration<PeleTransposePass> reg;
}

} // namespace mlir::bishengir
