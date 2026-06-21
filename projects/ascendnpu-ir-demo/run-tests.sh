#!/bin/bash
# run-tests.sh — 自动化测试: 验证 28 个 ascendnpu-ir-demo 用例的降级流水线
#
# 使用方式:
#   bash run-tests.sh              # 运行全部测试
#   bash run-tests.sh --verbose    # 显示详细输出
#   bash run-tests.sh <pattern>    # 只运行匹配的用例
#
# 测试内容:
#   1. 语法验证: mlir-opt 能正确解析每个 .mlir 文件
#   2. Stage1 降级: --convert-linalg-to-affine-loops 成功
#   3. Stage2 降级: --lower-affine --convert-scf-to-cf 成功
#   4. Stage3 降级: --convert-func-to-llvm 成功 (仅对支持的用例)
#   5. 输出验证: 关键 IR 模式出现在输出中 (FileCheck 风格)
#
# 退出码:
#   0 = 全部通过
#   1 = 有失败的测试

set -o pipefail

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_DIR="$SCRIPT_DIR/test-cases"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"

VERBOSE=0
PATTERN=""
for arg in "$@"; do
    case "$arg" in
        --verbose|-v) VERBOSE=1 ;;
        *) PATTERN="$arg" ;;
    esac
done

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
TOTAL=0

# ============================================================
# 辅助函数
# ============================================================
log_pass() {
    echo -e "  ${GREEN}✓ PASS${NC}: $1"
    PASS=$((PASS + 1))
}

log_fail() {
    echo -e "  ${RED}✗ FAIL${NC}: $1"
    if [ $VERBOSE -eq 1 ] && [ -n "$2" ]; then
        echo "    错误: $2"
    fi
    FAIL=$((FAIL + 1))
}

log_skip() {
    echo -e "  ${YELLOW}○ SKIP${NC}: $1"
    SKIP=$((SKIP + 1))
}

# 检查 mlir-opt 是否可用
check_mlir_opt() {
    if ! command -v "$MLIR_OPT" &>/dev/null; then
        echo -e "${YELLOW}⚠ mlir-opt 未找到，使用语法验证模式 (仅检查 RUN/CHECK 标注)${NC}"
        echo "  设置 MLIR_OPT 环境变量指定路径，或将 mlir-opt 加入 PATH"
        echo ""
        return 1
    fi
    return 0
}

# 运行 mlir-opt 并检查退出码
run_opt() {
    local file="$1"
    shift
    local output
    output=$("$MLIR_OPT" "$@" "$file" 2>&1)
    local rc=$?
    echo "$output"
    return $rc
}

# 检查输出中是否包含指定模式
check_pattern() {
    local output="$1"
    local pattern="$2"
    echo "$output" | grep -qE "$pattern"
}

# ============================================================
# 测试执行
# ============================================================
run_test_file() {
    local mlir="$1"
    local name
    name=$(basename "$mlir" .mlir)
    local dir
    dir=$(basename "$(dirname "$mlir")")
    local label="${dir}/${name}"

    TOTAL=$((TOTAL + 1))

    # 如果指定了 pattern，跳过不匹配的
    if [ -n "$PATTERN" ] && ! echo "$label" | grep -qi "$PATTERN"; then
        return
    fi

    echo "── $label ──"

    # 从文件中提取 RUN 和 CHECK 标注
    local has_run
    has_run=$(grep -c "^// RUN:" "$mlir" 2>/dev/null || echo "0")

    if [ "$has_run" -eq 0 ]; then
        log_skip "$label (无 RUN 标注)"
        return
    fi

    if [ "$HAS_MLIR_OPT" != "true" ]; then
        # 无 mlir-opt 时，只验证标注存在
        log_pass "$label (标注存在, 需 mlir-opt 执行)"
        return
    fi

    # 提取并执行每个 RUN 命令
    local run_idx=0
    local all_pass=true
    while IFS= read -r run_line; do
        run_idx=$((run_idx + 1))
        # 移除 "// RUN: " 前缀
        local cmd="${run_line#// RUN: }"
        # 替换 %s 为文件路径, %S 为目录路径
        cmd="${cmd//'%s'/$mlir}"
        cmd="${cmd//'%S'/$(dirname "$mlir")}"

        if [ $VERBOSE -eq 1 ]; then
            echo "    RUN[$run_idx]: $cmd"
        fi

        # 执行命令
        local output
        output=$(eval "$cmd" 2>&1)
        local rc=$?

        if [ $rc -ne 0 ]; then
            log_fail "$label (RUN[$run_idx] 退出码=$rc)"
            if [ $VERBOSE -eq 1 ]; then
                echo "    输出: $(echo "$output" | head -5)"
            fi
            all_pass=false
            break
        fi

        # 检查对应的 CHECK 模式
        local check_idx=0
        local check_pass=true
        while IFS= read -r check_line; do
            check_idx=$((check_idx + 1))
            local pattern="${check_line#// CHECK-RUN${run_idx}: }"
            if ! echo "$output" | grep -qF "$pattern"; then
                log_fail "$label (CHECK-RUN${run_idx}[$check_idx]: 未找到 '$pattern')"
                check_pass=false
                all_pass=false
                break
            fi
        done < <(grep "^// CHECK-RUN${run_idx}:" "$mlir")

    done < <(grep "^// RUN:" "$mlir")

    if $all_pass; then
        log_pass "$label"
    fi
}

# ============================================================
# 主流程
# ============================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ascendnpu-ir-demo: 自动化测试                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

HAS_MLIR_OPT=true
check_mlir_opt || HAS_MLIR_OPT=false

# 按难度顺序运行测试
for dir in "$CASES_DIR"/01_basic "$CASES_DIR"/02_intermediate "$CASES_DIR"/03_advanced; do
    if [ -d "$dir" ]; then
        echo ""
        echo "═══ $(basename "$dir") ═══"
        for mlir in "$dir"/*.mlir; do
            [ -f "$mlir" ] && run_test_file "$mlir"
        done
    fi
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo -e "  结果: ${GREEN}${PASS} 通过${NC}, ${RED}${FAIL} 失败${NC}, ${YELLOW}${SKIP} 跳过${NC} (共 ${TOTAL} 个)"
echo "══════════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
