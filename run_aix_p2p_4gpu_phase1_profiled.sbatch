#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-4gpu-phase1
#SBATCH --account=thes2181
#SBATCH --time=01:30:00
#SBATCH --output=aix-p2p-4gpu-phase1-%j.out

#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
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
    export AIX_DIAGNOSTICS_DIR="$benchmark_dir/aix_diagnostics_4gpu_phase1_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_TIMELINE_DIR="$benchmark_dir/aix_p2p_timeline_4gpu_phase1_${SLURM_JOB_ID}_${case_name}"
    export AIX_P2P_RANGE_PIPELINE_DEPTH=3
    export AIX_P2P_INITIAL_CREDITS=1

    srun --no-kill --het-group=0 --ntasks=96 --cpus-per-task=1 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$cpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 19 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_4gpu_phase1_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_4gpu_phase1_${SLURM_JOB_ID}_${case_name}.csv" \
        : --het-group=1 --ntasks=4 --cpus-per-task=24 --cpu-bind=cores \
        --export=ALL,LD_LIBRARY_PATH="$gpu_node_ld_library_path" \
        "$benchmark_binary" --model "$model_path" --samples-per-rank "$samples_per_rank" \
        --batch-size "$batch_size" --input-shape "$input_shape" --output-shape "$output_shape" \
        --warmup-steps 1 --measured-steps 19 --input-seed 42 \
        --raw-output-prefix "$benchmark_dir/aix_output_4gpu_phase1_${SLURM_JOB_ID}_${case_name}" \
        --output "$benchmark_dir/aix_p2p_steps_4gpu_phase1_${SLURM_JOB_ID}_${case_name}.csv"
}

collective_binary="$benchmark_dir/build-collective/aix_p2p_benchmark"
pipelined_binary="$benchmark_dir/build-pipelined/aix_p2p_benchmark"
collective_prefix="$root_dir/AIxeleratorService-p2p/INSTALL-BASELINE"
pipelined_prefix="$root_dir/AIxeleratorService-pipelined/INSTALL-PIPELINED"
model_dir="$root_dir/mini_app/train_models/model_a"
mmcp_model="/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"

# 4 GPU Phase 1: 8 profiled cases (1 warmup + 19 warm ML steps = 20 steps total)
run_case perfect collective "$model_dir/perfect_cuda.pt" 1296000 4000000 "1296000,18" "1296000" "$collective_binary" "$collective_prefix"
run_case perfect pipelined "$model_dir/perfect_cuda.pt" 1296000 4000000 "1296000,18" "1296000" "$pipelined_binary" "$pipelined_prefix"
run_case giant collective "$model_dir/giant_cuda.pt" 12960 4000000 "12960,18" "12960" "$collective_binary" "$collective_prefix"
run_case giant pipelined "$model_dir/giant_cuda.pt" 12960 4000000 "12960,18" "12960" "$pipelined_binary" "$pipelined_prefix"
run_case watercnn collective "$model_dir/watercnn_cuda.pt" 1041600 4000000 "1041600,18" "1041600" "$collective_binary" "$collective_prefix"
run_case watercnn pipelined "$model_dir/watercnn_cuda.pt" 1041600 4000000 "1041600,18" "1041600" "$pipelined_binary" "$pipelined_prefix"
run_case mmcp collective "$mmcp_model" 3456 50000 "3456,5,512" "3456,2,512" "$collective_binary" "$collective_prefix"
run_case mmcp pipelined "$mmcp_model" 3456 50000 "3456,5,512" "3456,2,512" "$pipelined_binary" "$pipelined_prefix"
