#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p
#SBATCH --account=thes2181
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=aix-p2p-%j.out

set -euo pipefail

benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}" 
root_dir="$(realpath "$benchmark_dir/..")"
model="${AIX_P2P_MODEL:-$root_dir/mini_app/train_models/model_a/perfect_cuda.pt}"
aix_prefix="${AIXELERATOR_INSTALL_PREFIX:?Set AIXELERATOR_INSTALL_PREFIX to the selected baseline or pipelined AIx install.}"

source "$root_dir/set_env_claix23_cuda12.4.sh"
export AIX_COMMUNICATION_MODE="${AIX_COMMUNICATION_MODE:-collective}"
build_variant="$AIX_COMMUNICATION_MODE"
if [[ "${WITH_SCOREP:-OFF}" == "ON" ]]; then
    source "$root_dir/CPP-ML-Interface/env_scorep.sh"
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=mpi --io=none --memory=malloc --thread=none --nocuda"
    export CXX=scorep-mpicxx
    export SCOREP_ENABLE_PROFILING="${SCOREP_ENABLE_PROFILING:-true}"
    export SCOREP_ENABLE_TRACING="${SCOREP_ENABLE_TRACING:-false}"
    export SCOREP_TOTAL_MEMORY="${SCOREP_TOTAL_MEMORY:-16000K}"
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY:-$SLURM_SUBMIT_DIR/scorep_aix_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}}"
    export SCOREP_METRIC_PAPI=""
    build_variant+="-scorep"
fi
build_dir="$benchmark_dir/build-$build_variant"
cmake -S "$benchmark_dir" -B "$build_dir" \
    -DAIXELERATOR_INSTALL_PREFIX="$aix_prefix" \
    -DWITH_SCOREP="${WITH_SCOREP:-OFF}"
cmake --build "$build_dir" --parallel

export AIX_DIAGNOSTICS="${AIX_DIAGNOSTICS:-1}"
export AIX_DIAGNOSTICS_DIR="${AIX_DIAGNOSTICS_DIR:-$SLURM_SUBMIT_DIR/aix_diagnostics_${SLURM_JOB_ID}}"
export AIX_P2P_TIMELINE_DIR="${AIX_P2P_TIMELINE_DIR:-$SLURM_SUBMIT_DIR/aix_p2p_timeline_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}}"

srun --cpu-bind=cores "$build_dir/aix_p2p_benchmark" \
    --model "$model" \
    --batch-size "${AIX_P2P_BATCH_SIZE:-4000000}" \
    --warmup-steps "${AIX_P2P_WARMUP_STEPS:-1}" \
    --measured-steps "${AIX_P2P_MEASURED_STEPS:-10}" \
    --output "$SLURM_SUBMIT_DIR/aix_p2p_steps_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}.csv"
