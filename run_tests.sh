#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — run automated tests before every commit / deploy
#
# Usage:
#   bash run_tests.sh          # normal run
#   bash run_tests.sh --cov    # with coverage report
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

pass() { echo -e "${GREEN}✔  $*${RESET}"; }
fail() { echo -e "${RED}✘  $*${RESET}"; }
info() { echo -e "${CYAN}▸  $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠  $*${RESET}"; }

# ── Check dependencies ────────────────────────────────────────────────────────
info "Checking dependencies..."

if ! python -c "import pytest" 2>/dev/null; then
    warn "pytest not found — installing..."
    pip install pytest pytest-cov --quiet
fi

# ── Arguments ────────────────────────────────────────────────────────────────
COVERAGE=false
for arg in "$@"; do
    [[ "$arg" == "--cov" ]] && COVERAGE=true
done

# ── Run tests ────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}  VoiceCount — Automated Tests${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

PYTEST_ARGS=(
    "tests/"
    "-v"
    "--tb=short"
    "--no-header"
    "-p" "no:warnings"
)

if $COVERAGE; then
    PYTEST_ARGS+=(
        "--cov=core"
        "--cov=features/utils"
        "--cov=features/manage_orders"
        "--cov=features/create_order"
        "--cov-report=term-missing"
        "--cov-report=html:htmlcov"
    )
fi

set +e
python -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
set -e

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

if [ $EXIT_CODE -eq 0 ]; then
    pass "All tests passed."
    echo ""
    echo -e "  ${GREEN}Safe to deploy.${RESET} Next step:"
    echo -e "  ${YELLOW}→ Open CHECKLIST.md and run the 5-minute manual smoke tests.${RESET}"
    if $COVERAGE; then
        echo ""
        echo -e "  Coverage report: ${CYAN}htmlcov/index.html${RESET}"
    fi
else
    fail "Tests FAILED (exit $EXIT_CODE)."
    echo ""
    echo -e "  ${RED}Do NOT deploy until all tests pass.${RESET}"
    echo -e "  Fix the failures above, then re-run: ${CYAN}bash run_tests.sh${RESET}"
fi

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

exit $EXIT_CODE
