#!/usr/bin/env bash
#SBATCH --job-name=aix-input-staged
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=logs/aix-input-staged-%j.out

# Staged AIx input-assembly isolation: pinned MPI, H2D, H2D+synthetic compute,
# and H2D+compute+D2H. CUDA is used only to reproduce the AIx range pipeline.
set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT_DIR="$(realpath "${BENCHMARK_DIR}/..")"
BUILD_DIR="${BENCHMARK_DIR}/build-input-assembly"
RESULTS_DIR="${BENCHMARK_DIR}/input_assembly_results"
PAYLOAD_BYTES="${PAYLOAD_BYTES_ENV:-6220800}"
WARMUPS="${WARMUPS_ENV:-20}"
ITERATIONS="${ITERATIONS_ENV:-100}"
PIPELINE_DEPTH="${PIPELINE_DEPTH_ENV:-3}"
COMPUTE_REPEATS="${COMPUTE_REPEATS_ENV:-40000}"
SCENARIOS="${SCENARIOS_ENV:-24c1:96c4}"
MODES="${MODES_ENV:-collective:p2p-credit1:p2p-fullcredits}"
STAGES="${STAGES_ENV:-pinned-mpi:h2d:h2d-compute:full}"
MODELS="${MODELS_ENV:-mock}"
ASYNC_VALUES="${ASYNC_ENV:-0}"
MPI_THREAD_VALUES="${MPI_THREAD_ENV:-multiple}"
SUBMIT_CPU_US_VALUES="${SUBMIT_CPU_US_ENV:-0}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS_ENV:-90}"
SKIP_GENERIC_PLOT="${SKIP_GENERIC_PLOT_ENV:-0}"
TORCH_CMAKE_PREFIX="${TORCH_CMAKE_PREFIX_ENV:-}"

cd "${BENCHMARK_DIR}"
mkdir -p "${RESULTS_DIR}" logs
source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"
unset SCOREP_ENABLE_PROFILING SCOREP_ENABLE_TRACING SCOREP_EXPERIMENT_DIRECTORY
unset AIX_COMMUNICATION_MODE AIX_P2P_INITIAL_CREDITS AIX_P2P_TIMELINE_DIR
# Match the production solver's single-threaded CPU policy.
export OMP_NUM_THREADS=1
export TORCH_NUM_THREADS=1
# Bypass Torch's GPU auto-detection try_run during CMake configuration.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

if [[ -z "${TORCH_CMAKE_PREFIX}" ]]; then
    PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
    TORCH_CMAKE_PREFIX="$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || true)"
fi

CMAKEFLAGS=()
if [[ -n "${TORCH_CMAKE_PREFIX}" ]]; then
    CMAKEFLAGS+=("-DWITH_TORCH_MODEL=ON" "-DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX}")
fi
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release "${CMAKEFLAGS[@]}"
cmake --build "${BUILD_DIR}" --target aix_input_assembly_staged --parallel 8
BINARY="${BUILD_DIR}/aix_input_assembly_staged"

IFS=':' read -r -a scenarios <<< "${SCENARIOS}"
IFS=':' read -r -a modes <<< "${MODES}"
IFS=':' read -r -a stages <<< "${STAGES}"
IFS=':' read -r -a models <<< "${MODELS}"
# Slurm uses commas to separate --export entries; use ':' for all list-valued inputs.
IFS=':' read -r -a async_values <<< "${ASYNC_VALUES}"
IFS=':' read -r -a mpi_thread_values <<< "${MPI_THREAD_VALUES}"
IFS=':' read -r -a submit_cpu_us_values <<< "${SUBMIT_CPU_US_VALUES}"
HUNG_CASES="${RESULTS_DIR}/hung_cases_${SLURM_JOB_ID}.txt"
: > "${HUNG_CASES}"

for scenario in "${scenarios[@]}"; do
    ranks=24
    [[ "${scenario}" == "96c4" ]] && ranks=96
    for mode in "${modes[@]}"; do
        for mpi_thread in "${mpi_thread_values[@]}"; do
            for async_controller in "${async_values[@]}"; do
                if [[ "${mpi_thread}" == "single" && "${async_controller}" == "1" ]]; then
                    continue
                fi
                for submit_cpu_us in "${submit_cpu_us_values[@]}"; do
                    for model in "${models[@]}"; do
                        for stage in "${stages[@]}"; do
                            if [[ "${model}" != "mock" && "${stage}" == "pinned-mpi" ]]; then
                                continue
                            fi
                            model_arg="${model}"
                            if [[ "${model}" != "mock" && "${model}" != /* ]]; then
                                model_arg="${ROOT_DIR}/mini_app/train_models/model_a/${model}"
                                [[ "${model_arg}" == *.pt ]] || model_arg="${model_arg}.pt"
                            fi
                            model_tag="$(basename "${model}" .pt)"
                            output="${RESULTS_DIR}/staged_${scenario}_${mode}_${stage}_thread${mpi_thread}_async${async_controller}_cpu${submit_cpu_us}_${model_tag}_${SLURM_JOB_ID}.csv"
                            echo "================================================================="
                            echo "scenario=${scenario}, ranks=${ranks}, mode=${mode}, thread=${mpi_thread}, async=${async_controller}, cpu=${submit_cpu_us}, model=${model_tag}, stage=${stage}"
                            echo "================================================================="
                            if timeout --foreground "${CASE_TIMEOUT_SECONDS}" srun --ntasks="${ranks}" --cpus-per-task=1 --distribution=block:block --cpu-bind=cores \
                                "${BINARY}" \
                                --mode "${mode}" \
                                --stage "${stage}" \
                                --group-size 24 \
                                --payload-bytes "${PAYLOAD_BYTES}" \
                                --warmups "${WARMUPS}" \
                                --iterations "${ITERATIONS}" \
                                --pipeline-depth "${PIPELINE_DEPTH}" \
                                --compute-repeats "${COMPUTE_REPEATS}" \
                                --mpi-thread "${mpi_thread}" \
                                --async-controller "${async_controller}" \
                                --submit-cpu-us "${submit_cpu_us}" \
                                --model "${model_arg}" \
                                --output "${output}"; then
                                echo "Completed: ${scenario},${mode},thread=${mpi_thread},async=${async_controller},cpu=${submit_cpu_us},model=${model_tag},${stage}"
                            else
                                status=$?
                                echo "FAILED_OR_TIMED_OUT status=${status}: ${scenario},${mode},thread=${mpi_thread},async=${async_controller},cpu=${submit_cpu_us},model=${model_tag},${stage}" | tee -a "${HUNG_CASES}"
                            fi
                        done
                    done
                done
            done
        done
    done
done

if [[ "${SKIP_GENERIC_PLOT}" != "1" ]]; then
    python3 plot_staged_input_assembly.py \
        --input-glob "${RESULTS_DIR}/staged_*_${SLURM_JOB_ID}.csv" \
        --output-dir "${RESULTS_DIR}" \
        --prefix "staged_${SLURM_JOB_ID}"
fi
