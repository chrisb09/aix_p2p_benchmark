#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-render
#SBATCH --account=thes2181
#SBATCH --partition=c23ms
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=aix-p2p-render-%j.out

set -euo pipefail

benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}" 
suite_job_id="${AIX_SUITE_JOB_ID:?Set AIX_SUITE_JOB_ID to the suite Slurm job ID.}"

for model_name in perfect giant watercnn; do
    python3 "$benchmark_dir/plot_p2p_timeline.py" \
        "$benchmark_dir/aix_p2p_timeline_suite_${suite_job_id}_${model_name}_pipelined"
done
