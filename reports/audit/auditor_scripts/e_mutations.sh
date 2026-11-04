#!/bin/zsh
# Audit script E (B11): mutation-style check. Copies src/ehrforecast and tests/ to a scratch
# directory (the repository is not touched), applies one mutation per run, and runs the boundary
# tests against the mutated copy via PYTHONPATH. The repo's own copy is never imported because
# PYTHONPATH precedes the editable-install .pth entry on sys.path.
set -u
REPO=<repo>
SCR="$(cd "$(dirname "$0")" && pwd)"
MUT="$SCR/mutant"
PY="$REPO/.venv/bin/python"
TESTS="tests/test_truncation.py tests/test_labels.py tests/test_cohort.py tests/test_audit_boundaries.py tests/test_history_boundary.py"
export PYTHONDONTWRITEBYTECODE=1

fresh() {
  rm -rf "$MUT"; mkdir -p "$MUT/src"
  cp -R "$REPO/src/ehrforecast" "$MUT/src/ehrforecast"
  cp -R "$REPO/tests" "$MUT/tests"
  find "$MUT" -name __pycache__ -prune -exec rm -rf {} +
  rm -rf "$MUT/src/ehrforecast.egg-info"
}

run() {  # $1 label
  echo "=================== $1 ==================="
  (cd "$MUT" && PYTHONPATH="$MUT/src" "$PY" -c "import ehrforecast, sys; print('ehrforecast from', ehrforecast.__file__)" \
    && PYTHONPATH="$MUT/src" "$PY" -m pytest -q -p no:cacheprovider -x --no-header ${=TESTS} 2>&1 | tail -15)
}

fresh; run "M0 unmutated copy (expect all pass)"

fresh
sed -i '' 's/keep = df\[spec.time_col\] < anchor$/keep = df[spec.time_col] <= anchor/' "$MUT/src/ehrforecast/truncation.py"
grep -n "keep = df" "$MUT/src/ehrforecast/truncation.py"
run "M1 truncation: strict < anchor -> inclusive <= (expect failures)"

fresh
sed -i '' 's/after = out\[spec.stop_col\] >= anchor\[keep\]/after = out[spec.stop_col] > anchor[keep]/' "$MUT/src/ehrforecast/truncation.py"
grep -n "after = out" "$MUT/src/ehrforecast/truncation.py"
run "M2 truncation: blank STOP >= anchor -> STOP > anchor (expect failures)"

fresh
sed -i '' 's/in_window = (first\["first_date"\] >= anchor)/in_window = (first["first_date"] > anchor)/' "$MUT/src/ehrforecast/labels.py"
grep -n "in_window" "$MUT/src/ehrforecast/labels.py"
run "M3 labels: window start inclusive -> exclusive (expect failures)"

fresh
sed -i '' 's/(first\["first_date"\] <= window_end)/(first["first_date"] < window_end)/' "$MUT/src/ehrforecast/labels.py"
grep -n "in_window" "$MUT/src/ehrforecast/labels.py"
run "M4 labels: window end inclusive -> exclusive (expect failures)"

fresh
sed -i '' 's/\.min()$/.max()/' "$MUT/src/ehrforecast/labels.py"
grep -n "max()" "$MUT/src/ehrforecast/labels.py"
run "M5 labels: first-ever diagnosis -> latest diagnosis (expect failures)"

fresh
sed -i '' 's/HORIZON = pd.DateOffset(years=HORIZON_YEARS)/HORIZON = pd.Timedelta(days=1826)/' "$MUT/src/ehrforecast/cohort.py"
grep -n "^HORIZON" "$MUT/src/ehrforecast/cohort.py"
run "M6 cohort: calendar years -> 1826 days (expect failures)"

rm -rf "$MUT"
