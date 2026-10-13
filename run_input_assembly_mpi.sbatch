#!/usr/bin/env bash
#SBATCH --job-name=aix-input-assembly
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/aix-input-assembly-%j.out

# Pure-MPI comparison of AIx's input assembly patterns on an exclusive 96-core
# c23mm node. This fixture never loads AIx or uses CUDA.
set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(realpath "${BENCHMARK_DIR}/..")"
RESULTS_DIR="${BENCHMARK_DIR}/input_assembly_results"
BUILD_DIR="${BENCHMARK_DIR}/build-input-assembly"
PAYLOAD_BYTES=6220800  # 320 x 270 x 18 FP32 values: one 96c4gpu WaterCNN rank input.
WARMUPS=20
ITERATIONS=200

cd "${BENCHMARK_DIR}"
mkdir -p "${RESULTS_DIR}" logs
source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"

# Do not accidentally inherit instrumentation or AIx runtime behavior.
unset SCOREP_ENABLE_PROFILING SCOREP_ENABLE_TRACING SCOREP_EXPERIMENT_DIRECTORY
unset AIX_COMMUNICATION_MODE AIX_P2P_INITIAL_CREDITS AIX_P2P_TIMELINE_DIR
unset AIX_P2P_ASYNC_CONTROLLER AIX_P2P_RANGE_PIPELINE AIX_P2P_RANGE_PIPELINE_DEPTH

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel 8
BINARY="${BUILD_DIR}/aix_input_assembly_mpi"

run_case() {
    local scenario="$1"
    local ranks="$2"
    local mode="$3"
    local output="${RESULTS_DIR}/input_assembly_${scenario}_${mode}_${SLURM_JOB_ID}.csv"

    echo "================================================================="
    echo "Scenario=${scenario}, ranks=${ranks}, mode=${mode}, payload=${PAYLOAD_BYTES} B/rank"
    echo "================================================================="
    srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
        "${BINARY}" \
        --mode "${mode}" \
        --group-size 24 \
        --payload-bytes "${PAYLOAD_BYTES}" \
        --warmups "${WARMUPS}" \
        --iterations "${ITERATIONS}" \
        --output "${output}"
}

# One 24-rank workgroup, then four independent workgroups concurrently.
for scenario in 24c1 96c4; do
    ranks=24
    if [[ "${scenario}" == "96c4" ]]; then
        ranks=96
    fi
    run_case "${scenario}" "${ranks}" collective
    run_case "${scenario}" "${ranks}" p2p-credit1
    run_case "${scenario}" "${ranks}" p2p-fullcredits
done

python3 plot_input_assembly.py \
    --input-glob "${RESULTS_DIR}/input_assembly_*_${SLURM_JOB_ID}.csv" \
    --output-dir "${RESULTS_DIR}" \
    --prefix "input_assembly_${SLURM_JOB_ID}"
