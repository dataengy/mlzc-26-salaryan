#!/usr/bin/env bash
# Diagnostics and optional fail-fast pipeline run for the salary predictor.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mode="diagnostics"
log_file=""
for arg in "$@"; do
  case "$arg" in
    --pipeline) mode="pipeline" ;;
    --help|-h)
      cat <<'USAGE'
Usage: scripts/doctor/doctor.sh [--pipeline] [LOG_FILE]

Default: gather diagnostics without downloading data, training, or modifying files.
--pipeline: run `make data`, `make eda`, `make train`, and `make test` in order,
            stopping immediately if any command fails.
LOG_FILE:   tee the complete output to this path as well as the terminal.
USAGE
      exit 0
      ;;
    *)
      if [[ -z "$log_file" ]]; then log_file="$arg"; else echo "Unexpected argument: $arg" >&2; exit 2; fi
      ;;
  esac
done

timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date)"
if [[ -n "$log_file" ]]; then
  mkdir -p "$(dirname "$log_file")"
  set -o pipefail
  exec > >(tee "$log_file") 2>&1
fi

printf '%s\n' '=== Salary project doctor ===' "mode=$mode" "time_utc=$timestamp" "project=$ROOT" "host=$(uname -srm 2>/dev/null || echo unknown)"

if [[ "$mode" == "pipeline" ]]; then
  for target in data eda train test; do
    printf '\n=== RUN make %s ===\n' "$target"
    if make "$target"; then
      printf '=== PASS make %s ===\n' "$target"
    else
      status=$?
      printf '=== FAIL make %s (exit %s); stopping ===\n' "$target" "$status" >&2
      exit "$status"
    fi
  done
  printf '\n%s\n' '=== Pipeline passed: data, eda, train, test ==='
  exit 0
fi

printf 'shell=%s\n' "${SHELL:-unknown}"
printf 'PATH=%s\n' "$PATH"

section() { printf '\n--- %s ---\n' "$1"; }
check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf '%s=' "$name"
    command -v "$name"
    case "$name" in
      uv|python|python3|pytest|direnv|make|docker) "$name" --version 2>&1 | head -n 1 || true ;;
    esac
  else
    printf '%s=NOT_FOUND\n' "$name"
  fi
}

section 'Commands'
for c in direnv uv python python3 pytest make docker curl; do check_cmd "$c"; done

section 'Virtual environment and packages'
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python - <<'PY'
import importlib.metadata as md
import sys
print(f"venv_python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
for package in ("numpy", "pandas", "scikit-learn", "joblib", "fastapi", "uvicorn", "matplotlib", "pytest", "httpx"):
    try:
        print(f"package_{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"package_{package}=MISSING")
PY
else
  printf '%s\n' 'venv_python=MISSING (.venv/bin/python not executable)'
fi

section 'Environment activation'
if command -v direnv >/dev/null 2>&1; then
  direnv status 2>&1 | sed -n '1,24p' || true
fi
printf 'envrc_exists='; [[ -f .envrc ]] && echo yes || echo no
printf 'venv_bin_in_PATH='; case ":$PATH:" in *":$ROOT/.venv/bin:"*) echo yes;; *) echo no;; esac

section 'Expected files'
for file in requirements.txt Makefile .envrc scripts/download.py src/mlzc_26_salaryan/eda.py src/mlzc_26_salaryan/train.py src/mlzc_26_salaryan/app.py src/mlzc_26_salaryan/model_utils.py notebooks/salary-prediction.ipynb data/examples/sample-request.json data/salaries.csv data/source.json reports/EDA.md reports/salary_histogram.png models/model.joblib models/metrics.json; do
  if [[ -f "$file" ]]; then printf 'file=%s exists bytes=%s\n' "$file" "$(wc -c < "$file" | tr -d ' ')"; else printf 'file=%s MISSING\n' "$file"; fi
done

section 'Dataset validation'
if [[ -f data/salaries.csv && -x .venv/bin/python ]]; then
  .venv/bin/python - <<'PY'
from pathlib import Path
try:
    import pandas as pd
    path = Path("data/salaries.csv")
    df = pd.read_csv(path)
    required = {"experience_level", "employment_type", "job_title", "employee_residence", "remote_ratio", "company_location", "company_size", "salary_in_usd", "work_year"}
    print(f"dataset_rows={len(df)} dataset_columns={len(df.columns)}")
    print(f"required_columns_missing={sorted(required - set(df.columns))}")
    if "salary_in_usd" in df:
        print(f"target_non_null={int(df.salary_in_usd.notna().sum())}")
    if "work_year" in df:
        print(f"work_year_counts={df.work_year.value_counts(dropna=False).sort_index().to_dict()}")
except Exception as exc:
    print(f"dataset_read_error={type(exc).__name__}: {exc}")
PY
else
  echo 'dataset_validation=SKIPPED (dataset or virtualenv missing)'
fi

section 'Network and source URL'
if command -v curl >/dev/null 2>&1; then
  curl -ILsS --connect-timeout 5 --max-time 10 -o /dev/null -w 'source_http_status=%{http_code}\n' https://raw.githubusercontent.com/foorilla/ai-jobs-net-salaries/main/salaries.csv 2>&1 || true
else
  echo 'network_check=SKIPPED (curl missing)'
fi

section 'Tests'
if [[ -x .venv/bin/python && -f tests/test_pipeline.py ]]; then
  .venv/bin/python -m pytest -q 2>&1
  printf 'pytest_exit=%s\n' "$?"
else
  echo 'pytest=SKIPPED (virtualenv or tests missing)'
fi

section 'API checks'
if command -v curl >/dev/null 2>&1; then
  curl -sS --connect-timeout 2 --max-time 4 -w '\napi_health_http_status=%{http_code}\n' http://127.0.0.1:8080/health 2>&1 || true
  if [[ -f data/examples/sample-request.json ]]; then
    curl -sS --connect-timeout 2 --max-time 8 -w '\napi_predict_http_status=%{http_code}\n' -H 'Content-Type: application/json' --data-binary @data/examples/sample-request.json http://127.0.0.1:8080/predict 2>&1 || true
  fi
fi

section 'Git state'
if command -v git >/dev/null 2>&1; then
  git status --short --branch 2>&1 || true
  git diff --check 2>&1 || true
fi

printf '\n%s\n' '=== Doctor complete ===' 'Copy the complete output above and send it back. Default diagnostics mode does not download data, install packages, or train a model.'
