#!/usr/bin/env bash
#SBATCH --job-name=aix-p2p-postprocess
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=aix-p2p-postprocess-%j.out

set -euo pipefail

benchmark_dir="${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")")}" 
suite_job_id="${AIX_SUITE_JOB_ID:?Set AIX_SUITE_JOB_ID to the suite Slurm job ID.}"
prefix_type="${AIX_PREFIX_TYPE:-full}"

echo "=== Output Equivalence Verification ==="
for model_name in watercnn mmcp perfect giant; do
    coll_json="$benchmark_dir/aix_output_${prefix_type}_${suite_job_id}_${model_name}_collective.json"
    pipe_json="$benchmark_dir/aix_output_${prefix_type}_${suite_job_id}_${model_name}_pipelined.json"
    if [[ -f "$coll_json" && -f "$pipe_json" ]]; then
        python3 "$benchmark_dir/compare_aix_outputs.py" "$coll_json" "$pipe_json" \
            --report "$benchmark_dir/aix_compare_${prefix_type}_${suite_job_id}_${model_name}.json"
    fi
done

echo "=== Timeline Rendering ==="
for model_name in watercnn mmcp perfect giant; do
    timeline_dir="$benchmark_dir/aix_p2p_timeline_${prefix_type}_${suite_job_id}_${model_name}_pipelined"
    if [[ -d "$timeline_dir" ]]; then
        python3 "$benchmark_dir/plot_p2p_timeline.py" "$timeline_dir"
    fi
done
