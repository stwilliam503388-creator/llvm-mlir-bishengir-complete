#!/bin/bash
# run-demo.sh — 用 mlir-opt 模拟 bishengir 三阶段降级
set -e

export LLVM_DIR="/opt/homebrew/opt/llvm"
export PATH="$LLVM_DIR/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_DIR="$SCRIPT_DIR/test-cases"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"

LLVM_VERSION=$(mlir-opt --version 2>&1 | head -1 || echo "unknown")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  bishengir-demo: 模拟三阶段降级                            ║"
echo "║  MLIR: $LLVM_VERSION"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 定义流水线
# ============================================================

# bishengir 三阶段对照:
BISHENGIR_PIPELINE="convert-linalg-to-hfusion + convert-arith-to-hfusion + convert-hfusion-to-hivm"
# 标准 MLIR 等效:
STD_PIPELINE="convert-linalg-to-affine-loops + lower-affine + convert-scf-to-cf"

echo "bishengir 流水线:  $BISHENGIR_PIPELINE"
echo "标准 MLIR 流水线:  $STD_PIPELINE"
echo ""

# ============================================================
# 处理每个测试用例
# ============================================================
for mlir in "$CASES_DIR"/*.mlir; do
    name=$(basename "$mlir" .mlir)
    echo "──────────────────────────────────────────────────────────────"
    echo "  用例: $name"
    echo "──────────────────────────────────────────────────────────────"

    # Stage 0: 原始 IR
    echo ""
    echo "  [Stage 0] 原始 Linalg IR:"
    echo "    mlir-opt $name.mlir  (原样输出)"
    HEAD=$(head -5 "$mlir")
    echo "    $HEAD"
    echo "    (${name}.mlir, 共 $(wc -l < "$mlir") 行)"
    echo ""

    # Stage 1: Linalg → Affine (模拟 bishengir 的 LinalgToHFusion)
    echo "  [Stage 1] linalg → affine  (对应 bishengir -convert-linalg-to-hfusion)"
    STAGE1="$RESULT_DIR/${name}_stage1_affine.mlir"
    mlir-opt --convert-linalg-to-affine-loops "$mlir" > "$STAGE1" 2>&1
    ST1_LINES=$(wc -l < "$STAGE1")
    ST1_SIZE=$(wc -c < "$STAGE1")
    echo "    输出: $STAGE1 (${ST1_LINES}行, ${ST1_SIZE}字节)"

    # Stage 2: Lower Affine + SCF (模拟 bishengir 的 ArithToHFusion)
    echo "  [Stage 2] affine → scf → cf  (对应 bishengir -convert-arith-to-hfusion)"
    STAGE2="$RESULT_DIR/${name}_stage2_scf.mlir"
    mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf "$mlir" > "$STAGE2" 2>&1
    ST2_LINES=$(wc -l < "$STAGE2")
    ST2_SIZE=$(wc -c < "$STAGE2")
    echo "    输出: $STAGE2 (${ST2_LINES}行, ${ST2_SIZE}字节)"

    # Stage 3: LLVM Dialect (模拟 bishengir 的 HFusionToHIVM 最终输出)
    echo "  [Stage 3] → LLVM IR  (对应 bishengir hivm.vadd/store 输出)"
    STAGE3="$RESULT_DIR/${name}_stage3_llvm.mlir"
    mlir-opt \
      --convert-linalg-to-affine-loops \
      --lower-affine \
      --convert-scf-to-cf \
      --convert-func-to-llvm \
      "$mlir" > "$STAGE3" 2>&1
    ST3_LINES=$(wc -l < "$STAGE3")
    ST3_SIZE=$(wc -c < "$STAGE3")
    echo "    输出: $STAGE3 (${ST3_LINES}行, ${ST3_SIZE}字节)"

    # 摘要: 每阶段的行数变化反映降级复杂度
    echo ""
    echo "  降级摘要:  原始 ${name}.mlir  →  $(wc -l < "$mlir")行"
    echo "              Stage1 (affine)   →  ${ST1_LINES}行"
    echo "              Stage2 (scf)      →  ${ST2_LINES}行"
    echo "              Stage3 (llvm)     →  ${ST3_LINES}行"
    echo ""
done

# ============================================================
# 对照表: bishengir vs 标准 MLIR
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  bishengir ↔ 标准 MLIR 对照表                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  bishengir 流水线:                   标准 MLIR 流水线:"
echo "  ─────────────────────────           ─────────────────────"
echo "  输入: Linalg IR                      输入: Linalg IR"
echo "    │                                     │"
echo "  Pass1: -convert-linalg-to-hfusion     Pass1: --convert-linalg-to-affine-loops"
echo "    Linalg → HFusion dialect              Linalg → affine dialect"
echo "    │                                     │"
echo "  Pass2: -convert-arith-to-hfusion      Pass2: --lower-affine"
echo "    Arith → HFusion dialect               affine → scf dialect"
echo "    │                                     │"
echo "  Pass3: -convert-hfusion-to-hivm       Pass3: --convert-scf-to-cf"
echo "    HFusion → HIVM dialect                scf → cf dialect"
echo "    │                                     │"
echo "  NPU IR (可通过 CANN 执行)             LLVM IR (可通过 lli 执行)"
echo ""

echo "结果保存在: $RESULT_DIR/"
ls -lh "$RESULT_DIR/"
