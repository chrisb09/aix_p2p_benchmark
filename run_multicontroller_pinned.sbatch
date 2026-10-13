#!/usr/bin/env bash
#SBATCH --job-name=aix-mc-pinned
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aix-mc-pinned-%j.out

# GPU-free controller-count sweep using the pure-MPI fixture: workgroup size is
# fixed at 6 ranks and the number of concurrent workgroups is 1/2/4/8. This
# isolates MPI progress interference between controllers with no GPU, no
# cgroup quota, and no shared-GPU confounds. Collective is included as a
# reference at each width.
set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(realpath "${BENCHMARK_DIR}/..")"
BINARY="${BENCHMARK_DIR}/build-input-assembly/aix_input_assembly_mpi"
RESULTS_DIR="${BENCHMARK_DIR}/input_assembly_results"

cd "${BENCHMARK_DIR}"
mkdir -p "${RESULTS_DIR}" logs
source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"

PAYLOAD="${PAYLOAD_BYTES_ENV:-6220800}"
WARMUPS="${WARMUPS_ENV:-10}"
ITERATIONS="${ITERATIONS_ENV:-100}"
GROUP="${GROUP_SIZE_ENV:-6}"

run_case() {
    local tag="$1" ranks="$2" mode="$3"
    local output="${RESULTS_DIR}/mcmpi_${tag}_${SLURM_JOB_ID}.csv"
    echo "================================================================="
    echo "case=${tag}, ranks=${ranks}, workgroups=$((ranks / GROUP)), group_size=${GROUP}, mode=${mode}"
    echo "================================================================="
    if timeout --foreground 120 srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
        "${BINARY}" --mode "${mode}" --group-size "${GROUP}" \
        --payload-bytes "${PAYLOAD}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}" \
        --output "${output}"; then
        echo "Completed: ${tag}"
    else
        echo "FAILED_OR_TIMED_OUT: ${tag}"
    fi
}

for width in 1 2 4 8; do
    ranks=$((width * GROUP))
    run_case "wg${width}_p2p" "${ranks}" p2p-credit1
    run_case "wg${width}_coll" "${ranks}" collective
done

echo ""
echo "Job ${SLURM_JOB_ID} finished."
