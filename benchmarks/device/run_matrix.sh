#!/usr/bin/env bash
# Measure several library versions on one device from the current checkout.
#
#   IGL_BENCH_DEVICE=<cpu|mps|cuda> benchmarks/device/run_matrix.sh <label>=<src-dir> [<label>=<src-dir> ...]
#
# Each <src-dir> is the ``src`` directory of a worktree (or the current tree);
# it is put first on PYTHONPATH so the harness in this checkout measures that
# version, and the results are filed under <label>
# (``results/benchmarks/device/<label>/<device>/``). Optional environment:
#   PY      python executable (default: python)
#   SUITE   space-separated modules to run (default: regions syncs components profile e2e)
#   EXTRA   extra arguments passed to every module (e.g. "--problems small")
set -euo pipefail
PY=${PY:-python}
DEV=${IGL_BENCH_DEVICE:-cpu}
SUITE=${SUITE:-"regions syncs components profile e2e"}
EXTRA=${EXTRA:-}
[ $# -ge 1 ] || { echo "usage: $0 <label>=<src-dir> [...]" >&2; exit 2; }
for spec in "$@"; do
  label=${spec%%=*}; src=${spec#*=}
  [ -d "$src/igl" ] || { echo "no igl package under $src" >&2; exit 2; }
  export PYTHONPATH="$src${PYTHONPATH:+:$PYTHONPATH}" IGL_BENCH_LABEL="$label" IGL_BENCH_DEVICE="$DEV"
  found=$($PY -c 'import igl, os; print(os.path.dirname(os.path.abspath(igl.__file__)))')
  [ "$found" = "$(cd "$src/igl" && pwd -P)" ] || { echo "PYTHONPATH did not take: igl loads from $found" >&2; exit 2; }
  echo "== $label ($src) on $DEV: $SUITE =="
  for mod in $SUITE; do
    # shellcheck disable=SC2086
    $PY -m "benchmarks.device.$mod" --no-gate $EXTRA || echo "FAILED: $label $mod on $DEV (exit $?)"
  done
  export PYTHONPATH="${PYTHONPATH#"$src"}"; PYTHONPATH="${PYTHONPATH#:}"
done
