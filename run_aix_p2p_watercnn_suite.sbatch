#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-suite
#SBATCH --account=thes2181
#SBATCH --time=00:40:00
#SBATCH --output=aix-p2p-suite-%j.out

#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

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
source "$root_dir/set_env_claix23_cuda12.4.sh"
source "$root_dir/CPP-ML-Interface/env_scorep.sh"

export SCOREP_ENABLE_TRACING=true
export SCOREP_ENABLE_PROFILING=false
export SCOREP_TOTAL_MEMORY=1G
export SCOREP_THREAD_MODE=pthread
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true

cpu_node_ld_library_path="$CUDA_ROOT/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH:-}"
gpu_node_ld_library_path=""
IFS=":" read -r -a ld_library_paths <<< "${LD_LIBRARY_PATH:-}"
for library_path in "${ld_library_paths[@]}"; do
    if [[ "$library_path" != */stubs ]]; then
        gpu_node_ld_library_path+="${gpu_node_ld_library_path:+:}$library_path"
    fi
done

run_case() {
    local model_name="$1"
    local communication_mode="$2"
    local model_path="$3"
    local samples_per_rank="$4"
    local benchmark_binary="$5"
    local aix_prefix="$6"
    local case_name="${model_name}_${communication_mode}"

    export AIX_COMMUNICATION_MODE="$communication_mode"
    export AIXELERATOR_INSTALL_PREFIX="$aix_prefix"
    export SCOREP_EXPERIMENT_DIRECTORY="$benchmark_dir/scorep_aix_het_suite_${SLURM_JOB_ID}_${case_name}"
    export AIX_DIAGNOSTICS=1
    export AIX_DIAGNOSTICS_DIR="$benchmark_dir/aix_diagnostics_suite_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_TIMELINE_DIR="$benchmark_dir/aix_p2p_timeline_suite_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_RANGE_PIPELINE_DEPTH=3

    srun --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$cpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size 4000000 --warmup-steps 1 --measured-steps 10 \
        --output "$benchmark_dir/aix_p2p_steps_suite_${SLURM_JOB_ID}_${case_name}.csv" \
        : --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$gpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size 4000000 --warmup-steps 1 --measured-steps 10 \
        --output "$benchmark_dir/aix_p2p_steps_suite_${SLURM_JOB_ID}_${case_name}.csv"
}

collective_binary="$benchmark_dir/build-prebuilt-collective-scorep-mpi/aix_p2p_benchmark"
pipelined_binary="$benchmark_dir/build-prebuilt-pipelined-scorep-mpi/aix_p2p_benchmark"
collective_prefix="$root_dir/AIxeleratorService-p2p/INSTALL-BASELINE-SCOREP-MPI"
pipelined_prefix="$root_dir/AIxeleratorService-pipelined/INSTALL-PIPELINED-SCOREP-MPI"
model_dir="$root_dir/mini_app/train_models/model_a"

run_case perfect collective "$model_dir/perfect_cuda.pt" 1296000 "$collective_binary" "$collective_prefix"
run_case perfect pipelined "$model_dir/perfect_cuda.pt" 1296000 "$pipelined_binary" "$pipelined_prefix"
run_case giant collective "$model_dir/giant_cuda.pt" 12960 "$collective_binary" "$collective_prefix"
run_case giant pipelined "$model_dir/giant_cuda.pt" 12960 "$pipelined_binary" "$pipelined_prefix"
run_case watercnn collective "$model_dir/watercnn_cuda.pt" 104160 "$collective_binary" "$collective_prefix"
run_case watercnn pipelined "$model_dir/watercnn_cuda.pt" 104160 "$pipelined_binary" "$pipelined_prefix"
