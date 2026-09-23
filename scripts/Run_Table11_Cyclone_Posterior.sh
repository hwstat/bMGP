#!/bin/bash
# Tables 6 and 12 of the paper: posterior comparison on the cyclone data. Captures the draws of
# QMP, bagged QMP, QMP-GP, bagged QMP-GP and DQP at the settings and seeds of
# Table 11, then reduces them to the slope in Year of the conditional quantile
# of lifetime maximum wind speed. Runs src/cyclone/.
# Usage: bash Run_Table11_Cyclone_Posterior.sh [step ...]
#
#   bash Run_Table11_Cyclone_Posterior.sh              # every step, in order
#   bash Run_Table11_Cyclone_Posterior.sh summarize    # only the steps named
#
# Steps: qmp, bqmp, bqmp_gp, dqp, summarize
#
# PYTHON picks the interpreter (default python3).
#
# The three posterior_* scripts re-implement nothing. Each imports the matching
# timing script, replaces the sampler name in its module namespace by a
# pass-through that keeps the returned array, and then calls that script's own
# main() with the command line Run_Table10_Cyclone_Timing.sh uses;
# posterior_dqp.R calls time_dqp.R with the argument that makes it save the
# chain. Their timing JSONs are by-products, since they also pay for writing
# the draws to disk; the reported timings are the ones from Table 11.
#
# posterior_qmp.py saves the GP draws as beta_gp in the same file, so QMP-GP
# needs no separate run. summarize_posteriors.py adds a bagged QMP-GP row when
# bqmp_gp_beta_draws.npz is there and skips it when it is not.
#
# Outputs go to scripts/output/cyclone/posterior: the saved draws, the run
# logs, cyclone_slope_summary.csv, cyclone_slope_checks.csv and
# cyclone_slope_table.tex. The whole sweep is about half an hour, most of it
# the bagged QMP run and the DQP chain.
#
# Thread policy, identical to Run_Table10_Cyclone_Timing.sh.

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
CYC="$ROOT/src/cyclone"
DATA="$ROOT/data/globalTCmax4.txt"
PY=${PYTHON:-python3}
POST="$ROOT/scripts/output/cyclone/posterior"
LOG="$ROOT/scripts/output/cyclone/logs"
mkdir -p "$POST" "$LOG"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

WANT="$*"
run_step () { [ -z "$WANT" ] || printf '%s\n' $WANT | grep -qx "$1"; }

if run_step qmp; then
  echo "=== QMP and QMP-GP draws: B=10000 paths, T=5000 steps ==="
  "$PY" "$CYC/posterior_qmp.py" --data "$DATA" --B 10000 --T 5000 \
      --out "$POST/qmp_beta_draws.npz" > "$LOG/posterior_qmp.log" 2>&1
fi

if run_step bqmp; then
  echo "=== bagged QMP draws: 50 resamples x 20 paths ==="
  "$PY" "$CYC/posterior_bqmp.py" --data "$DATA" --B-boot 50 --M-B 20 --T 5000 \
      --out "$POST/bqmp_beta_draws.npz" > "$LOG/posterior_bqmp.log" 2>&1
fi

if run_step bqmp_gp; then
  echo "=== bagged QMP-GP draws: 50 resamples x 20 GP draws ==="
  "$PY" "$CYC/posterior_bqmp.py" --data "$DATA" --B-boot 50 --M-B 20 --T 5000 \
      --sampler gp --out "$POST/bqmp_gp_beta_draws.npz" \
      > "$LOG/posterior_bqmp_gp.log" 2>&1
fi

if run_step dqp; then
  echo "=== DQP draws: 20000 MCMC iterations, seeded initial pyramid ==="
  Rscript "$CYC/posterior_dqp.R" 20000 "$POST" > "$LOG/posterior_dqp.log" 2>&1
fi

if run_step summarize; then
  echo "=== slope summary and the Table 12 body ==="
  "$PY" "$CYC/summarize_posteriors.py" --postdir "$POST"
fi
