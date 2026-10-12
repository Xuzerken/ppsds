#!/usr/bin/env bash
# Linux/WSL equivalent of run.ps1 (the .ps1 is kept for Windows users).
#
# Usage: ./run.sh [cpu] [seed] [model] [control]
#   cpu     CPU count       (default: 16)
#   seed    random seed     (default: 3)
#   model   model type      (default: final_round)
#             linear:    final_round / original / larger_x0 /
#                        different_Sigma_bar / different_system_convergence
#             nonlinear: NLink / higher_dim
#   control control type    (default: none)
#             `regulation` is only valid for the linear models; nonlinear models exit
#             with an error. The default is `none` because that is safe for **every**
#             model -- pass `regulation` explicitly to run it.
#
# Runs 6 experiments in sequence: random(p=100) / random(p=1) / worst, each for both
# `none` and $CONTROL.
# The trajectory count uses run.py's default (100).
#
# Requires the conda environment `drpp` (see environment.yml).
set -u

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drpp

CPU=${1:-16}
SEED=${2:-3}
MODEL=${3:-final_round}
CONTROL=${4:-none}

cd "$(dirname "$0")"

for C in none "$CONTROL"; do
    python run.py -n "$CPU" -s "$SEED" -p 100 -m "$MODEL" -c "$C"
    python run.py -n "$CPU" -s "$SEED" -p 1   -m "$MODEL" -c "$C"
    python run.py -n "$CPU" -s "$SEED"        -m "$MODEL" -c "$C" -e worst
done
