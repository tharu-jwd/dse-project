#!/usr/bin/env bash
# One command, everything runs, evidence lands in reports/<date>/.
# Every step is independent: one failing or missing prerequisite must
# never stop the others. See docs/TESTING_REPORT.md Section 4.2 for
# what the summary table feeds into.
set -uo pipefail
cd "$(dirname "$0")/.."

DATE="$(date +%F)"
OUT="reports/$DATE"
mkdir -p "$OUT"

SUMMARY="$OUT/summary.md"
echo "# Test run — $DATE" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| Step | Tests | Verdict | Seconds |" >> "$SUMMARY"
echo "|---|---|---|---|" >> "$SUMMARY"

_row() {
  # _row "<step>" "<tests>" "<verdict>" "<seconds>"
  echo "| $1 | $2 | $3 | $4 |" >> "$SUMMARY"
}

_count_junit() {
  # Prints "<passed>/<total>" from a JUnit XML, or "?" if unreadable.
  python3 - "$1" <<'PY' 2>/dev/null || echo "?"
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
    suites = root.findall(".//testsuite") or [root]
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    print(f"{total - failed}/{total}")
except Exception:
    print("?")
PY
}

# --- Backend --------------------------------------------------------

echo "[test-plan] backend suite"
start=$(date +%s)
if command -v docker > /dev/null 2>&1 && docker image inspect dse-project-backend:latest > /dev/null 2>&1; then
  # Runs inside the project's own backend image so this works the same
  # on any machine, regardless of what's installed in the local Python -
  # the deps (torch, silero-vad, ...) only ever live in that image.
  if docker run --rm --network host --env-file .env \
      -v "$PWD:/repo" -w /repo dse-project-backend:latest \
      pytest test/backend --junitxml="/repo/$OUT/backend-junit.xml" > "$OUT/backend.log" 2>&1; then
    verdict="PASS"
  else
    verdict="FAIL"
  fi
elif pytest test/backend --junitxml="$OUT/backend-junit.xml" > "$OUT/backend.log" 2>&1; then
  verdict="PASS"
else
  verdict="FAIL"
fi
secs=$(( $(date +%s) - start ))
_row "Backend (pytest)" "$(_count_junit "$OUT/backend-junit.xml")" "$verdict" "$secs"

# --- Frontend component tests ----------------------------------------

echo "[test-plan] frontend component tests"
if [ -d frontend/node_modules ]; then
  start=$(date +%s)
  if (cd frontend && npx vitest run --reporter=default > "../$OUT/frontend.log" 2>&1); then
    verdict="PASS"
  else
    verdict="FAIL"
  fi
  secs=$(( $(date +%s) - start ))
  tests=$(grep -oE '[0-9]+ passed' "$OUT/frontend.log" | tail -1 || echo "?")
  _row "Frontend (vitest)" "${tests:-?}" "$verdict" "$secs"
else
  echo "[test-plan] frontend/node_modules missing, skipping" | tee "$OUT/frontend.log"
  _row "Frontend (vitest)" "-" "SKIPPED (no node_modules)" "0"
fi

# --- Playwright e2e ----------------------------------------------------

echo "[test-plan] playwright e2e"
if [ -d frontend/node_modules ] && (cd frontend && npx playwright --version > /dev/null 2>&1); then
  start=$(date +%s)
  if (cd frontend && npx playwright test --reporter=junit > "../$OUT/playwright-junit.xml" 2> "../$OUT/playwright.log"); then
    verdict="PASS"
  else
    verdict="FAIL"
  fi
  secs=$(( $(date +%s) - start ))
  _row "Playwright (e2e)" "$(_count_junit "$OUT/playwright-junit.xml")" "$verdict" "$secs"
else
  echo "[test-plan] playwright not installed, skipping" | tee "$OUT/playwright.log"
  _row "Playwright (e2e)" "-" "SKIPPED (playwright not installed)" "0"
fi

# --- Deployment smoke tests (only if the live URL answers) ------------

echo "[test-plan] deployment smoke tests"
SMOKE_API_URL="${SMOKE_API_URL:-https://sinhaspeech.duckdns.org}"
if curl -sf --max-time 5 "$SMOKE_API_URL/health" > /dev/null 2>&1; then
  start=$(date +%s)
  if SMOKE_API_URL="$SMOKE_API_URL" SMOKE_APP_URL="${SMOKE_APP_URL:-https://sinhaspeech.vercel.app}" \
     pytest test/deployment --asyncio-mode=auto --junitxml="$OUT/deployment-junit.xml" > "$OUT/deployment.log" 2>&1; then
    verdict="PASS"
  else
    verdict="FAIL"
  fi
  secs=$(( $(date +%s) - start ))
  _row "Deployment smoke" "$(_count_junit "$OUT/deployment-junit.xml")" "$verdict" "$secs"
else
  echo "[test-plan] $SMOKE_API_URL/health did not answer, skipping" | tee "$OUT/deployment.log"
  _row "Deployment smoke" "-" "SKIPPED (live URL unreachable)" "0"
fi

echo ""
echo "[test-plan] done. Evidence in $OUT/"
cat "$SUMMARY"
