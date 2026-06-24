#!/bin/bash
# run-demo.sh — 用标准 mlir-opt 模拟 AscendNPU-IR 三阶段降级

# Intentionally omit `set -e`: run_stage callers use `|| true` so unsupported lowering
# stages can be logged while the demo continues. `-u` catches unset variables; `pipefail`
# propagates failures in pipelines.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_DIR="$SCRIPT_DIR/test-cases/mlir"
RESULT_DIR="$SCRIPT_DIR/results"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
mkdir -p "$RESULT_DIR"

if ! command -v "$MLIR_OPT" &>/dev/null; then
    echo "❌ mlir-opt 未找到。请安装 LLVM/MLIR，或设置 MLIR_OPT=/path/to/mlir-opt。"
    exit 1
fi

LLVM_VERSION=$($MLIR_OPT --version 2>&1 | head -1 || echo "unknown")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ascendnpu-ir-demo: 模拟三阶段降级                            ║"
echo "║  MLIR: $LLVM_VERSION"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "AscendNPU-IR 流水线:  convert-linalg-to-hfusion + convert-arith-to-hfusion + convert-hfusion-to-hivm"
echo "标准 MLIR 流水线:     convert-linalg-to-affine-loops + lower-affine + convert-scf-to-cf + convert-func-to-llvm"
echo ""

run_stage() {
    local label="$1"
    local out="$2"
    shift 2
    if "$MLIR_OPT" "$@" > "$out" 2>&1; then
        echo "    ✓ $label: $out ($(wc -l < "$out")行, $(wc -c < "$out")字节)"
        return 0
    fi
    echo "    ⚠ $label: 失败，日志已保存到 $out"
    return 1
}

while IFS= read -r mlir; do
    rel=${mlir#"$CASES_DIR"/}
    name=${rel%.mlir}
    # Use filesystem-safe names for result files derived from nested case paths.
    safe_name=${name//\//__}

    echo "──────────────────────────────────────────────────────────────"
    echo "  用例: $rel"
    echo "──────────────────────────────────────────────────────────────"
    echo "  [Stage 0] 原始 MLIR: $mlir ($(wc -l < "$mlir")行)"

    stage1="$RESULT_DIR/${safe_name}_stage1_affine.mlir"
    stage2="$RESULT_DIR/${safe_name}_stage2_cf.mlir"
    stage3="$RESULT_DIR/${safe_name}_stage3_llvm.mlir"

    echo "  [Stage 1] linalg → affine  (类比 -convert-linalg-to-hfusion)"
    run_stage "Stage1" "$stage1" --convert-linalg-to-affine-loops "$mlir" || true

    echo "  [Stage 2] affine → cf  (类比 -convert-arith-to-hfusion)"
    run_stage "Stage2" "$stage2" --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf "$mlir" || true

    echo "  [Stage 3] → LLVM dialect  (类比 -convert-hfusion-to-hivm 后继续 lowering)"
    run_stage "Stage3" "$stage3" --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-func-to-llvm "$mlir" || true
    echo ""
done < <(find "$CASES_DIR" -mindepth 2 -maxdepth 2 -name '*.mlir' | sort)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AscendNPU-IR ↔ 标准 MLIR 对照                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  AscendNPU-IR: Linalg → HFusion → HIVM → CANN/LLVM"
echo "  标准 MLIR:    Linalg → Affine/SCF/CF → LLVM dialect"
echo ""
echo "结果保存在: $RESULT_DIR/"
ls -lh "$RESULT_DIR/"
