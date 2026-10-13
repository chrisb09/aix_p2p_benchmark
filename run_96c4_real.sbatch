#!/usr/bin/env bash
#SBATCH --job-name=aix-96c4-real
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aix-96c4-real-%j.out

# Scale-dependent GPU-pipeline test: identical per-controller work (24 ranks,
# same payload, same model) at 1 controller/1 GPU versus 4 controllers/4 GPUs,
# plus mock controls. Input-ready is the CPU/MPI metric of interest.
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
MODE="${MODE_ENV:-p2p-credit1}"

run_case() {
    local tag="$1" ranks="$2" group="$3" model="$4"
    local output="${RESULTS_DIR}/scale_${tag}_${SLURM_JOB_ID}.csv"
    echo "================================================================="
    echo "case=${tag}, ranks=${ranks}, group_size=${group}, workgroups=$((ranks / group)), model=$(basename "${model}")"
    echo "================================================================="
    if timeout --foreground 180 srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
        "${BINARY}" --mode "${MODE}" --stage full --group-size "${group}" \
        --payload-bytes "${PAYLOAD}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}" \
        --mpi-thread multiple --async-controller 0 --model "${model}" --output "${output}"; then
        echo "Completed: ${tag}"
    else
        echo "FAILED_OR_TIMED_OUT: ${tag}"
    fi
}

run_case 24c1_real 24 24 "${MODEL}"
run_case 24c1_mock 24 24 mock
run_case 96c4_real 96 24 "${MODEL}"
run_case 96c4_mock 96 24 mock

echo ""
echo "Job ${SLURM_JOB_ID} finished."
