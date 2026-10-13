#!/usr/bin/env bash
# regenerate_p2p_timelines.sh
# ============================
# Regenerate all tracked aix_p2p_benchmark timeline PNGs using the updated
# plot_p2p_timeline.py script (which now applies thesis_style fonts).
#
# Run from: /hpcwork/ro092286/smartsim/aix_p2p_benchmark/
#
# Source→output mapping (pipelined = P2P runs; job ID with most steps used):
#
#   plots_1gpu/                        ← 1gpu_phase1_2887779_watercnn_pipelined (20 steps → steps 1-19)
#   plots_4gpu/                        ← 4gpu_phase1_2887782_watercnn_pipelined (20 steps → steps 1-19)
#   plots_het_24plus24/giant/          ← het24_giant_mmcp_2951118_giant_pipelined (11 → 1-10)
#   plots_het_24plus24/mmcp/           ← het24_giant_mmcp_2951118_mmcp_pipelined (11 → 1-10)
#   plots_het_24plus24/perfect/        ← het24_2914415_perfect_pipelined (11 → 1-10)
#   plots_het_24plus24/transformer/    ← het24p24_transformer_3033094_transformer_pipelined (11 → 1-10)
#   plots_het_24plus24/watercnn/       ← het24_2914415_watercnn_pipelined (11 → 1-10)
#   plots_het_24plus1/transformer/     ← het24p1_transformer_3033092_transformer_pipelined (11 → 1-10)
#   plots_het_24plus1_scorep/transformer/ ← 1gpu_scorep_3033202_transformer_pipelined (? → 1-10)
#   plots_het_24plus24_scorep/transformer/ ← 4gpu_scorep_3033204_transformer_pipelined (? → 1-10)
#   plots_het_24plus24_210k/perfect/   ← het24_210k_3033096_perfect_pipelined (11 → 1-10)
#   plots_het_24plus24_210k/watercnn/  ← het24_210k_3033096_watercnn_pipelined (11 → 1-10)

set -euo pipefail

SCRIPT=/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/plot_p2p_timeline.py
PYTHON=/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3
BENCH=/hpcwork/ro092286/smartsim/aix_p2p_benchmark
DATA=/rwthfs/rz/cluster/home/ro092286/Master-Thesis/gitlab/data/aix_p2p_benchmark

render() {
    local src="$1" dst="$2" steps="$3" model="$4"
    echo "=== Rendering: $src → $dst ==="
    mkdir -p "$DATA/$dst"
    # Build --step flags
    step_args=()
    for s in $steps; do step_args+=("--step" "$s"); done
    $PYTHON "$SCRIPT" "$BENCH/$src" --output-dir "$DATA/$dst" --model "$model" "${step_args[@]}"
}

# 1-GPU pipelined (steps 1-19)
render "aix_p2p_timeline_1gpu_phase1_2887779_watercnn_pipelined" \
       "plots_1gpu" \
       "$(seq 1 19 | tr '\n' ' ')" "watercnn"

# 4-GPU pipelined (steps 1-19)
render "aix_p2p_timeline_4gpu_phase1_2887782_watercnn_pipelined" \
       "plots_4gpu" \
       "$(seq 1 19 | tr '\n' ' ')" "watercnn"

# het24+24 pipelined (steps 1-10)
render "aix_p2p_timeline_het24_giant_mmcp_2951118_giant_pipelined" \
       "plots_het_24plus24/giant" \
       "$(seq 1 10 | tr '\n' ' ')" "giant_mlp"

render "aix_p2p_timeline_het24_giant_mmcp_2951118_mmcp_pipelined" \
       "plots_het_24plus24/mmcp" \
       "$(seq 1 10 | tr '\n' ' ')" "mmcp_test_mlp_m5"

render "aix_p2p_timeline_het24_2914415_perfect_pipelined" \
       "plots_het_24plus24/perfect" \
       "$(seq 1 10 | tr '\n' ' ')" "perfect"

render "aix_p2p_timeline_het24p24_transformer_3033094_transformer_pipelined" \
       "plots_het_24plus24/transformer" \
       "$(seq 1 10 | tr '\n' ' ')" "tbl_transformer"

render "aix_p2p_timeline_het24_2914415_watercnn_pipelined" \
       "plots_het_24plus24/watercnn" \
       "$(seq 1 10 | tr '\n' ' ')" "watercnn"

# het24+1 pipelined (steps 1-10)
render "aix_p2p_timeline_het24p1_transformer_3033092_transformer_pipelined" \
       "plots_het_24plus1/transformer" \
       "$(seq 1 10 | tr '\n' ' ')" "tbl_transformer"

# ScoreP runs (steps 1-10)
render "aix_p2p_timeline_1gpu_scorep_3033202_transformer_pipelined" \
       "plots_het_24plus1_scorep/transformer" \
       "$(seq 1 10 | tr '\n' ' ')" "tbl_transformer"

render "aix_p2p_timeline_4gpu_scorep_3033204_transformer_pipelined" \
       "plots_het_24plus24_scorep/transformer" \
       "$(seq 1 10 | tr '\n' ' ')" "tbl_transformer"

# het24+24 210k (steps 1-10)
render "aix_p2p_timeline_het24_210k_3033096_perfect_pipelined" \
       "plots_het_24plus24_210k/perfect" \
       "$(seq 1 10 | tr '\n' ' ')" "perfect"

render "aix_p2p_timeline_het24_210k_3033096_watercnn_pipelined" \
       "plots_het_24plus24_210k/watercnn" \
       "$(seq 1 10 | tr '\n' ' ')" "watercnn"

echo "All timeline figures regenerated."
