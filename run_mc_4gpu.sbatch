#!/usr/bin/env bash
#SBATCH --job-name=aix-mc-4gpu
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aix-mc-4gpu-%j.out

# Small 4-GPU companion to run_multicontroller_small: fixed group size 6,
# 1 vs 4 controllers, each controller on its own GPU. Isolates concurrent
# real-pipeline interference from the shared-GPU artifact of job 3962840.
set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(realpath "${BENCHMARK_DIR}/..")"
BINARY="${BENCHMARK_DIR}/build-input-assembly/aix_input_assembly_staged"
MODEL="${ROOT_DIR}/mini_app/train_models/model_a/watercnn_cuda.pt"
RESULTS_DIR="${BENCHMARK_DIR}/input_assembly_results"

cd "${BENCHMARK_DIR}"
mkdir -p "${RESULTS_DIR}" logs
source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"
export OMP_NUM_THREADS=1
export TORCH_NUM_THREADS=1
unset AIX_COMMUNICATION_MODE AIX_P2P_INITIAL_CREDITS AIX_P2P_TIMELINE_DIR

PAYLOAD="${PAYLOAD_BYTES_ENV:-6220800}"
WARMUPS="${WARMUPS_ENV:-10}"
ITERATIONS="${ITERATIONS_ENV:-100}"
GROUP="${GROUP_SIZE_ENV:-6}"

run_case() {
    local tag="$1" ranks="$2" model="$3"
    local output="${RESULTS_DIR}/mc4gpu_${tag}_${SLURM_JOB_ID}.csv"
    echo "================================================================="
    echo "case=${tag}, ranks=${ranks}, workgroups=$((ranks / GROUP)), group_size=${GROUP}, model=$(basename "${model}")"
    echo "================================================================="
    if timeout --foreground 120 srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
        "${BINARY}" --mode p2p-credit1 --stage full --group-size "${GROUP}" \
        --payload-bytes "${PAYLOAD}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}" \
        --mpi-thread multiple --async-controller 0 --model "${model}" --output "${output}"; then
        echo "Completed: ${tag}"
    else
        echo "FAILED_OR_TIMED_OUT: ${tag}"
    fi
}

run_case wg1_real 6 "${MODEL}"
run_case wg4_real 24 "${MODEL}"
run_case wg1_mock 6 mock
run_case wg4_mock 24 mock

echo ""
echo "Job ${SLURM_JOB_ID} finished."
