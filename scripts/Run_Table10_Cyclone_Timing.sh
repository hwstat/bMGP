#!/bin/bash
# Table 11 of the paper: wall-clock time of one posterior for the conditional quantiles of
# lifetime maximum wind speed on the North Atlantic cyclone records, for QMP,
# bagged QMP, QMP-GP, bagged QMP-GP and DQP. Runs src/cyclone/.
# Usage: bash Run_Table10_Cyclone_Timing.sh [family ...]
#
#   bash Run_Table10_Cyclone_Timing.sh                  # every family, two reps each
#   bash Run_Table10_Cyclone_Timing.sh qmp_gp bqmp_gp   # only the families named
#
# Families: qmp, bqmp, qmp_small, qmp_gp, bqmp_gp, dqp
#
# PYTHON picks the interpreter (default python3) and REPS the repeat labels
# (default "1 2"), so a single pilot run is
#   REPS=1 bash Run_Table10_Cyclone_Timing.sh qmp_gp
#
# Outputs go to scripts/output/cyclone: one JSON per run under raw/, the run
# logs under logs/, then cyclone_timings.csv and the markdown and LaTeX tables.
#
# Nothing runs concurrently, because concurrent runs would contend for cores
# and the wall-clock numbers would stop meaning anything. The full sweep is
# about 35 minutes, most of it the two DQP chains.
#
# Thread policy, applied identically to the Python and the R processes:
#   OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1,
#   VECLIB_MAXIMUM_THREADS=1
#   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Each run records its own measured CPU-time / wall-time ratio, which is the
# number to read if you want to know how many cores the run actually used, and
# the one-minute load average before and after itself, which is what says
# whether the machine was busy. make_tables.py warns about rows whose load was
# high.

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
CYC="$ROOT/src/cyclone"
DATA="$ROOT/data/globalTCmax4.txt"
PY=${PYTHON:-python3}
OUT="$ROOT/scripts/output/cyclone"
RES="$OUT/raw"
LOG="$OUT/logs"
mkdir -p "$RES" "$LOG"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# One-minute load average, without assuming macOS: /proc/loadavg on Linux,
# uptime anywhere else. Printed for the log only; the numbers that matter are
# the per-run load fields in the JSONs.
load1 () {
  if [ -r /proc/loadavg ]; then
    cut -d' ' -f1 /proc/loadavg
  else
    uptime | sed 's/.*load averages*:[ ]*//' | awk '{print $1}' | tr -d ','
  fi
}

WANT="$*"
run_family () { [ -z "$WANT" ] || printf '%s\n' $WANT | grep -qx "$1"; }
REPS=${REPS:-"1 2"}

echo "starting with one-minute load average $(load1)"

for REP in $REPS; do
  if run_family qmp; then
    echo "=== [rep $REP] QMP, upstream settings: B=10000 paths, T=5000 steps ==="
    "$PY" "$CYC/time_qmp.py" --data "$DATA" --B 10000 --T 5000 --rep "$REP" \
        --out "$RES/qmp_B10000_rep${REP}.json" > "$LOG/qmp_B10000_rep${REP}.log" 2>&1
  fi

  if run_family bqmp; then
    echo "=== [rep $REP] bagged QMP, paper settings: 50 resamples x 20 paths, T=5000 ==="
    "$PY" "$CYC/time_bqmp.py" --data "$DATA" --B-boot 50 --M-B 20 --T 5000 --rep "$REP" \
        --out "$RES/bqmp_50x20_rep${REP}.json" > "$LOG/bqmp_50x20_rep${REP}.log" 2>&1
  fi

  if run_family qmp_small; then
    echo "=== [rep $REP] QMP at 1000 paths (matched-draw reference row) ==="
    "$PY" "$CYC/time_qmp.py" --data "$DATA" --B 1000 --T 5000 --rep "$REP" \
        --out "$RES/qmp_B1000_rep${REP}.json" > "$LOG/qmp_B1000_rep${REP}.log" 2>&1
  fi

  if run_family qmp_gp; then
    echo "=== [rep $REP] QMP-GP, upstream settings: approx_PR_reg_B, B=10000 draws, seed 5124 ==="
    "$PY" "$CYC/time_qmp.py" --data "$DATA" --B 10000 --T 5000 --rep "$REP" \
        --method gp \
        --out "$RES/qmpgp_B10000_rep${REP}.json" > "$LOG/qmpgp_B10000_rep${REP}.log" 2>&1
  fi

  if run_family bqmp_gp; then
    echo "=== [rep $REP] bagged QMP-GP: 50 resamples x 20 GP draws ==="
    "$PY" "$CYC/time_bqmp.py" --data "$DATA" --B-boot 50 --M-B 20 --T 5000 --rep "$REP" \
        --sampler gp \
        --out "$RES/bqmpgp_50x20_rep${REP}.json" > "$LOG/bqmpgp_50x20_rep${REP}.log" 2>&1
  fi
done

if run_family dqp; then
  for REP in $REPS; do
    echo "=== [rep $REP] DQP, upstream MCMC settings: 20000 iterations ==="
    Rscript "$CYC/time_dqp.R" 20000 "$REP" "$RES/dqp_20000_rep${REP}.json" \
        "$CYC/dqp" "$DATA" \
        > "$LOG/dqp_20000_rep${REP}.log" 2>&1
  done
fi

echo "=== all runs finished, one-minute load average $(load1) ==="
"$PY" "$CYC/collect_timings.py" --raw "$RES" --out "$OUT/cyclone_timings.csv"
"$PY" "$CYC/make_tables.py" --csv "$OUT/cyclone_timings.csv" \
    --md-out "$OUT/timing_table.md" --tex-out "$OUT/timing_table.tex"
