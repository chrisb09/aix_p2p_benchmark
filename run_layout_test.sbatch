#!/usr/bin/env bash
#SBATCH --job-name=aix-layout
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=00:10:00
#SBATCH --output=logs/aix-layout-%j.out

# Rank-layout test: contiguous workgroups (rank / 24) versus production-style
# strided workgroups (world_rank % 4), where each controller's workers are
# spread across all NUMA domains. Same payload and real WaterCNN model.
set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(realpath "${BENCHMARK_DIR}/..")"
BINARY="${BENCHMARK_DIR}/build-input-assembly/aix_input_assembly_staged"
MODEL="${ROOT_DIR}/mini_app/train_models/model_a/watercnn_cuda.pt"
RESULTS_DIR="${BENCHMARK_DIR}/input_assembly_results"

cd "${BENCHMARK_DIR}"
mkdir -p "${RESULTS_DIR}" logs
# The shared env script reads module variables that may be unset on some
# partitions (e.g. EBROOTCLANG on c25g); relax nounset only while sourcing.
set +u
source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"
set -u
export OMP_NUM_THREADS=1
export TORCH_NUM_THREADS=1
unset AIX_COMMUNICATION_MODE AIX_P2P_INITIAL_CREDITS AIX_P2P_TIMELINE_DIR

PAYLOAD="${PAYLOAD_BYTES_ENV:-6220800}"
WARMUPS="${WARMUPS_ENV:-10}"
ITERATIONS="${ITERATIONS_ENV:-100}"

run_case() {
    local tag="$1" ranks="$2" group="$3" layout="$4" model="$5"
    local output="${RESULTS_DIR}/layout_${tag}_${SLURM_JOB_ID}.csv"
    echo "================================================================="
    echo "case=${tag}, ranks=${ranks}, group_size=${group}, layout=${layout}, model=$(basename "${model}")"
    echo "================================================================="
    if timeout --foreground 180 srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
        "${BINARY}" --mode p2p-credit1 --stage full --group-size "${group}" --group-layout "${layout}" \
        --payload-bytes "${PAYLOAD}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}" \
        --mpi-thread multiple --async-controller 0 --model "${model}" --output "${output}"; then
        echo "Completed: ${tag}"
    else
        echo "FAILED_OR_TIMED_OUT: ${tag}"
    fi
}

run_case 24c1_real 24 24 contiguous "${MODEL}"
run_case 96c4_contig_real 96 24 contiguous "${MODEL}"
run_case 96c4_strided_real 96 24 strided "${MODEL}"
run_case 96c4_contig_mock 96 24 contiguous mock
run_case 96c4_strided_mock 96 24 strided mock

echo ""
echo "Job ${SLURM_JOB_ID} finished."
