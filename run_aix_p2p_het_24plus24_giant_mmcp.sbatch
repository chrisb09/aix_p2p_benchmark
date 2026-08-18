#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-het-giant-mmcp
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --output=aix-p2p-het-giant-mmcp-%j.out

# Component 0: 24 solver ranks on a CPU-only node (c23mm).
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

# Component 1: 24 solver ranks on a GPU node (c23g) with 4 GPUs.
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

set -euo pipefail

benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}"
root_dir="$(realpath "$benchmark_dir/..")"
source "$root_dir/set_env_claix23_cuda12.4.sh"

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
    export AIX_DIAGNOSTICS_DIR="$benchmark_dir/aix_diagnostics_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_TIMELINE_DIR="$benchmark_dir/aix_p2p_timeline_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_RANGE_PIPELINE_DEPTH=3
    export AIX_P2P_INITIAL_CREDITS=1

    echo "=== Running ${case_name} (24 CPU ranks + 24 GPU ranks, 4 GPUs, batch=${batch_size}) ==="
    srun --wait=60 --no-kill --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$cpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 10 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}.csv" \
        : --wait=60 --no-kill --het-group=1 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$gpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 10 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_het24_giant_mmcp_${SLURM_JOB_ID}_${case_name}.csv"

    echo "=== Rendering timeline for ${case_name} ==="
    python3 "$benchmark_dir/plot_p2p_timeline.py" "$AIX_P2P_TIMELINE_DIR"
}

collective_binary="$benchmark_dir/build-collective/aix_p2p_benchmark"
pipelined_binary="$benchmark_dir/build-pipelined/aix_p2p_benchmark"
collective_prefix="$root_dir/CPP-ML-Interface/extern/AIxeleratorService/INSTALL"
pipelined_prefix="$root_dir/CPP-ML-Interface/extern/AIxeleratorService/INSTALL"
model_dir="$root_dir/mini_app/train_models/model_a"
mmcp_model="/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"

# Keep the prior model-specific inputs and batch sizes; only the implementation
# and renderer differ, so the resulting timelines can be compared directly.
run_case giant collective "$model_dir/giant_cuda.pt" 12960 4000000 "12960,18" "12960" "$collective_binary" "$collective_prefix"
run_case giant pipelined "$model_dir/giant_cuda.pt" 12960 4000000 "12960,18" "12960" "$pipelined_binary" "$pipelined_prefix"
run_case mmcp collective "$mmcp_model" 3456 50000 "3456,5,512" "3456,2,512" "$collective_binary" "$collective_prefix"
run_case mmcp pipelined "$mmcp_model" 3456 50000 "3456,5,512" "3456,2,512" "$pipelined_binary" "$pipelined_prefix"
