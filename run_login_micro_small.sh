#!/usr/bin/env bash
# Login-node micro reproduction, budget-respecting variant.
#
# Each busy-polling MPI process costs roughly one core on this node, and the
# user.slice quota is 4 cores. Use only 2-4 processes total so the cgroup does
# not throttle, then compare per-controller input-ready across 1 vs 2
# concurrent workgroups.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "${BENCH_DIR}/..")"
BINARY="${BENCH_DIR}/build-input-assembly/aix_input_assembly_staged"
MODEL="${ROOT_DIR}/mini_app/train_models/model_a/watercnn_cuda.pt"
RESULTS_DIR="${BENCH_DIR}/input_assembly_results"
CPU_STAT="/sys/fs/cgroup/user.slice/user-$(id -u).slice/cpu.stat"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ENV:-1,3}"
export OMP_NUM_THREADS=1
export TORCH_NUM_THREADS=1

source "${ROOT_DIR}/set_env_claix23_cuda12.4.sh"

PAYLOAD="${PAYLOAD_BYTES:-6220800}"
WARMUPS="${WARMUPS:-10}"
ITERATIONS="${ITERATIONS:-200}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/login_small_${STAMP}_cpu.csv"
echo "case,wall_s,usage_usec_delta,cores_avg,nr_throttled_delta,throttled_usec_delta" > "${LOG}"

read_cpu_stat() {
  awk '/^usage_usec/{u=$2} /^nr_throttled/{t=$2} /^throttled_usec/{tu=$2} END{print u+0, t+0, tu+0}' "${CPU_STAT}"
}

run_case() {
  local tag="$1" ranks="$2" group="$3" model="$4"
  local out="${RESULTS_DIR}/login_${tag}_${STAMP}.csv"
  echo "================================================================="
  echo "case=${tag}, ranks=${ranks}, group_size=${group}, model=$(basename "${model}")"
  echo "================================================================="
  local before after start end
  before="$(read_cpu_stat)"
  start="$(date +%s.%N)"
  mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
    -mca mpi_yield_when_idle 1 --bind-to none -np "${ranks}" \
    "${BINARY}" --mode p2p-credit1 --stage full --group-size "${group}" \
    --payload-bytes "${PAYLOAD}" --warmups "${WARMUPS}" --iterations "${ITERATIONS}" \
    --mpi-thread multiple --async-controller 0 --model "${model}" --output "${out}"
  end="$(date +%s.%N)"
  after="$(read_cpu_stat)"
  awk -v tag="${tag}" -v start="${start}" -v end="${end}" -v b="${before}" -v a="${after}" 'BEGIN{
    split(b,bb," "); split(a,aa," ");
    wall=end-start; du=aa[1]-bb[1]; dt=aa[2]-bb[2]; dtu=aa[3]-bb[3];
    printf "%s,%.3f,%d,%.3f,%d,%d\n", tag, wall, du, du/1e6/wall, dt, dtu
  }' | tee -a "${LOG}"
}

run_case a2_real 2 2 "${MODEL}"
run_case b2_real 4 2 "${MODEL}"
run_case b2_mock 4 2 mock

echo ""
echo "CPU accounting log: ${LOG}"
echo "STAMP=${STAMP}"
