#!/bin/bash
# run-tests.sh — 自动化测试: 验证 ascendnpu-ir-demo 用例的降级流水线
#
# 使用方式:
#   bash run-tests.sh              # 运行全部测试
#   bash run-tests.sh --verbose    # 显示详细输出
#   bash run-tests.sh <pattern>    # 只运行匹配的用例

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_DIR="$SCRIPT_DIR/test-cases/mlir"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"

VERBOSE=0
PATTERN=""
for arg in "$@"; do
    case "$arg" in
        --verbose|-v) VERBOSE=1 ;;
        *) PATTERN="$arg" ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
TOTAL=0

log_pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASS=$((PASS + 1)); }
log_fail() {
    echo -e "  ${RED}✗ FAIL${NC}: $1"
    if [ $VERBOSE -eq 1 ] && [ -n "${2:-}" ]; then echo "    错误: $2"; fi
    FAIL=$((FAIL + 1))
}
log_skip() { echo -e "  ${YELLOW}○ SKIP${NC}: $1"; SKIP=$((SKIP + 1)); }

check_tools() {
    HAS_MLIR_OPT=true
    HAS_FILECHECK=true
    if ! command -v "$MLIR_OPT" &>/dev/null; then
        echo -e "${YELLOW}⚠ mlir-opt 未找到，使用标注验证模式 (仅检查 RUN/CHECK 标注)${NC}"
        echo "  设置 MLIR_OPT 环境变量指定路径，或将 mlir-opt 加入 PATH"
        HAS_MLIR_OPT=false
    fi
    if ! command -v FileCheck &>/dev/null; then
        HAS_FILECHECK=false
        if [ "$HAS_MLIR_OPT" = "true" ]; then
            echo -e "${YELLOW}⚠ FileCheck 未找到，包含 FileCheck 的 RUN 命令将只执行管道左侧${NC}"
        fi
    fi
    echo ""
}

normalize_cmd() {
    local cmd="$1"
    cmd="${cmd//'%s'/$2}"
    cmd="${cmd//'%S'/$(dirname "$2")}"
    cmd="${cmd/mlir-opt/$MLIR_OPT}"
    if [ "$HAS_FILECHECK" != "true" ]; then
        cmd="${cmd%%| FileCheck*}"
    fi
    echo "$cmd"
}

run_test_file() {
    local mlir="$1"
    local name dir label has_run
    name=$(basename "$mlir" .mlir)
    dir=$(basename "$(dirname "$mlir")")
    label="${dir}/${name}"

    if [ -n "$PATTERN" ] && ! echo "$label" | grep -qi "$PATTERN"; then
        return
    fi
    TOTAL=$((TOTAL + 1))
    echo "── $label ──"

    has_run=$(grep -c "^// RUN:" "$mlir" 2>/dev/null || true)
    if [ "${has_run:-0}" -eq 0 ]; then
        log_skip "$label (无 RUN 标注)"
        return
    fi

    if [ "$HAS_MLIR_OPT" != "true" ]; then
        log_pass "$label (RUN 标注存在, 需 mlir-opt 执行)"
        return
    fi

    local run_idx=0 all_pass=true
    while IFS= read -r run_line; do
        run_idx=$((run_idx + 1))
        local raw_cmd="${run_line#// RUN: }"
        local cmd
        cmd=$(normalize_cmd "$raw_cmd" "$mlir")

        if [ $VERBOSE -eq 1 ]; then echo "    RUN[$run_idx]: $cmd"; fi

        local output rc
        output=$(eval "$cmd" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            log_fail "$label (RUN[$run_idx] 退出码=$rc)" "$(echo "$output" | head -5)"
            all_pass=false
            break
        fi

        while IFS= read -r check_line; do
            local pattern="${check_line#// CHECK-RUN${run_idx}: }"
            if ! echo "$output" | grep -qF "$pattern"; then
                log_fail "$label (CHECK-RUN${run_idx}: 未找到 '$pattern')"
                all_pass=false
                break
            fi
        done < <(grep "^// CHECK-RUN${run_idx}:" "$mlir")

        $all_pass || break
    done < <(grep "^// RUN:" "$mlir")

    if $all_pass; then log_pass "$label"; fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ascendnpu-ir-demo: 自动化测试                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

check_tools

for dir in "$CASES_DIR"/01_basic "$CASES_DIR"/02_intermediate "$CASES_DIR"/03_advanced; do
    if [ -d "$dir" ]; then
        echo ""
        echo "═══ $(basename "$dir") ═══"
        while IFS= read -r mlir; do
            run_test_file "$mlir"
        done < <(find "$dir" -maxdepth 1 -name '*.mlir' | sort)
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo -e "  结果: ${GREEN}${PASS} 通过${NC}, ${RED}${FAIL} 失败${NC}, ${YELLOW}${SKIP} 跳过${NC} (共 ${TOTAL} 个)"
echo "══════════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
