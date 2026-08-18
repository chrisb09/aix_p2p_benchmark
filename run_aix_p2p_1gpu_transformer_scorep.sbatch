#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-1gpu-trans-scorep
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --output=aix-p2p-1gpu-trans-scorep-%j.out

# Component 0: 24 solver ranks on a CPU-only node (c23mm)
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

# Component 1: 1 GPU controller rank on a GPU node (c23g)
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

export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_TOTAL_MEMORY=64M
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
    local batch_size="$5"
    local input_shape="$6"
    local output_shape="$7"
    local benchmark_binary="$8"
    local aix_prefix="$9"
    local case_name="${model_name}_${communication_mode}"

    export AIX_COMMUNICATION_MODE="$communication_mode"
    export AIXELERATOR_INSTALL_PREFIX="$aix_prefix"
    export AIX_MEASURE_MEMORY=1
    export AIX_DIAGNOSTICS=1
    export AIX_DIAGNOSTICS_DIR="$benchmark_dir/aix_diagnostics_1gpu_scorep_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_TIMELINE_DIR="$benchmark_dir/aix_p2p_timeline_1gpu_scorep_${SLURM_JOB_ID}_${case_name}"
    export SCOREP_EXPERIMENT_DIRECTORY="$benchmark_dir/scorep_aix_1gpu_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_RANGE_PIPELINE_DEPTH=3
    export AIX_P2P_INITIAL_CREDITS=1

    echo "=== Running ${case_name} with Score-P (24 CPU ranks + 1 GPU rank, 1 GPU, batch=${batch_size}) ==="
    srun --wait=60 --no-kill --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$cpu_node_ld_library_path",SCOREP_EXPERIMENT_DIRECTORY="$SCOREP_EXPERIMENT_DIRECTORY" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 10 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_1gpu_scorep_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_1gpu_scorep_${SLURM_JOB_ID}_${case_name}.csv" \
        : --wait=60 --no-kill --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$gpu_node_ld_library_path",SCOREP_EXPERIMENT_DIRECTORY="$SCOREP_EXPERIMENT_DIRECTORY" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 10 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_1gpu_scorep_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_1gpu_scorep_${SLURM_JOB_ID}_${case_name}.csv"

    echo "=== Rendering timeline for ${case_name} ==="
    python3 "$benchmark_dir/plot_p2p_timeline.py" "$AIX_P2P_TIMELINE_DIR"
}

scorep_binary="$benchmark_dir/build-scorep/aix_p2p_benchmark"
scorep_prefix="$root_dir/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-SCOREP"
model_dir="$root_dir/mini_app/train_models/model_a"

# Run Transformer with Score-P in both collective and pipelined modes
run_case transformer collective "$model_dir/transformer_cuda.pt" 104160 210000 "104160,18" "104160" "$scorep_binary" "$scorep_prefix"
run_case transformer pipelined "$model_dir/transformer_cuda.pt" 104160 210000 "104160,18" "104160" "$scorep_binary" "$scorep_prefix"
