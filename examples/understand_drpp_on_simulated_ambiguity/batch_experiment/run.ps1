# Windows equivalent of run.sh.
#
# Usage: .\run.ps1 [-cpu 16] [-seed 3] [-model final_round] [-control none]
#   model   linear:    final_round / original / larger_x0 /
#                      different_Sigma_bar / different_system_convergence
#           nonlinear: NLink / higher_dim
#   control `regulation` is only valid for the linear models; nonlinear models exit with
#           an error. The default is `none` because that is safe for **every** model.
#
# Runs 6 experiments in sequence: random(p=100) / random(p=1) / worst, each for both
# `none` and $control.
# The trajectory count uses run.py's default (100).
#
# Requires the conda environment `drpp` (see examples/environment.yml).
param(
    [int]$cpu=16,
    [int]$seed=3,
    [string]$model='final_round',
    [string]$control='none'
)
conda activate drpp

foreach ($c in @('none', $control)) {
    python .\run.py -n $cpu -s $seed -p 100 -m $model -c $c
    python .\run.py -n $cpu -s $seed -p 1   -m $model -c $c
    python .\run.py -n $cpu -s $seed        -m $model -c $c -e worst
}
