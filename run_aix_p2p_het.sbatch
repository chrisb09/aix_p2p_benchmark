#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-het
#SBATCH --account=thes2181
#SBATCH --time=00:20:00
#SBATCH --output=aix-p2p-het-%j.out

# Component 0: solver ranks on a CPU-only node.
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

# Component 1: one GPU controller rank. c23g requires the 24-core task allocation.
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

set -euo pipefail

benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}" 
root_dir="$(realpath "$benchmark_dir/..")"
model="${AIX_P2P_MODEL:-$root_dir/mini_app/train_models/model_a/perfect_cuda.pt}"
aix_prefix="${AIXELERATOR_INSTALL_PREFIX:?Set AIXELERATOR_INSTALL_PREFIX to the selected baseline or pipelined AIx install.}"
benchmark_binary="${AIX_P2P_BINARY:-}"

source "$root_dir/set_env_claix23_cuda12.4.sh"
export AIX_COMMUNICATION_MODE="${AIX_COMMUNICATION_MODE:-collective}"
build_variant="het-$AIX_COMMUNICATION_MODE"
if [[ "${WITH_SCOREP:-OFF}" == "ON" ]]; then
    source "$root_dir/CPP-ML-Interface/env_scorep.sh"
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=mpi --io=none --memory=malloc --thread=${SCOREP_THREAD_MODE:-pthread} --nocuda"
    export CXX=scorep-mpicxx
    export SCOREP_ENABLE_PROFILING="${SCOREP_ENABLE_PROFILING:-true}"
    export SCOREP_ENABLE_TRACING="${SCOREP_ENABLE_TRACING:-false}"
    export SCOREP_TOTAL_MEMORY="${SCOREP_TOTAL_MEMORY:-16000K}"
    export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY="${SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY:-true}"
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY:-$SLURM_SUBMIT_DIR/scorep_aix_het_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}}"
    export SCOREP_METRIC_PAPI=""
    build_variant+="-scorep"
fi
if [[ -z "$benchmark_binary" ]]; then
    build_dir="$benchmark_dir/build-$build_variant-${SLURM_JOB_ID:-local}"
    cmake -S "$benchmark_dir" -B "$build_dir" \
        -DAIXELERATOR_INSTALL_PREFIX="$aix_prefix" \
        -DWITH_SCOREP="${WITH_SCOREP:-OFF}"
    cmake --build "$build_dir" --parallel
    benchmark_binary="$build_dir/aix_p2p_benchmark"
elif [[ ! -x "$benchmark_binary" ]]; then
    echo "AIX_P2P_BINARY is not executable: $benchmark_binary" >&2
    exit 2
fi

export AIX_DIAGNOSTICS="${AIX_DIAGNOSTICS:-1}"
export AIX_DIAGNOSTICS_DIR="${AIX_DIAGNOSTICS_DIR:-$SLURM_SUBMIT_DIR/aix_diagnostics_het_${SLURM_JOB_ID}}"
export AIX_P2P_TIMELINE_DIR="${AIX_P2P_TIMELINE_DIR:-$SLURM_SUBMIT_DIR/aix_p2p_timeline_het_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}}"
cpu_node_ld_library_path="$CUDA_ROOT/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH:-}"
gpu_node_ld_library_path=""
IFS=":" read -r -a ld_library_paths <<< "${LD_LIBRARY_PATH:-}"
for library_path in "${ld_library_paths[@]}"; do
    if [[ "$library_path" == */stubs ]]; then
        continue
    fi
    gpu_node_ld_library_path+="${gpu_node_ld_library_path:+:}$library_path"
done

benchmark_args=(
    --model "$model"
    --samples-per-rank "${AIX_P2P_SAMPLES_PER_RANK:-129600}"
    --batch-size "${AIX_P2P_BATCH_SIZE:-4000000}"
    --warmup-steps "${AIX_P2P_WARMUP_STEPS:-1}"
    --measured-steps "${AIX_P2P_MEASURED_STEPS:-10}"
    --output "$SLURM_SUBMIT_DIR/aix_p2p_steps_het_${AIX_COMMUNICATION_MODE}_${SLURM_JOB_ID}.csv"
)

srun --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores --export=ALL,LD_LIBRARY_PATH="$cpu_node_ld_library_path" \
    "$benchmark_binary" "${benchmark_args[@]}" \
    : --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores --export=ALL,LD_LIBRARY_PATH="$gpu_node_ld_library_path" \
    "$benchmark_binary" "${benchmark_args[@]}"
