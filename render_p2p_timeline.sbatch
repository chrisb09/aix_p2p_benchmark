#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-render
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:15:00
#SBATCH --output=aix-p2p-render-%j.out

set -euo pipefail

: "${AIX_P2P_TIMELINE_DIR:?Set AIX_P2P_TIMELINE_DIR to a completed timeline directory.}"
benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}" 

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python3 "$benchmark_dir/plot_p2p_timeline.py" "$AIX_P2P_TIMELINE_DIR"
