# Analysis for Job 2823533 (aix-p2p-full-suite)

## Detailed Analysis

### Failure Summary
The Slurm job `aix-p2p-full-suite` (ID 2823533) failed with exit code 9:0 (SIGKILL). The job runs a full benchmark suite of 8 cases (4 models x 2 communication modes) using `run_aix_p2p_full_suite.sbatch`. Score-P tracing was enabled with `SCOREP_TOTAL_MEMORY=128000K` (~128 MB).

### Progression Through the Suite
The log shows the job successfully completed 3 of 4 cases before the 4th case (giant_pipelined) crashed:

1. **perfect_collective** — succeeded (line 7098: `AIX_P2P_RESULT output_valid=1`)
2. **perfect_pipelined** — succeeded (line 8255) with 2 Score-P trace buffer flush warnings on rank 24
3. **giant_collective** — succeeded (line 15852)
4. **giant_pipelined** — **FAILED** — Score-P out of memory on rank 24, causing SIGABRT and subsequent SIGKILL

Evidence of incomplete failure: `aix_p2p_steps_full_2823533_giant_pipelined.csv` is 0 bytes (empty), while `aix_p2p_steps_full_2823533_giant_collective.csv` is 7164 bytes.

### Root Cause: Score-P Tracing Memory Exhaustion
The failure occurs on rank 24 (node w23g0006, the GPU controller in the heterogeneous job).

**Score-P trace buffer exhaustion (lines 16505-16525):**
- Score-P repeatedly attempts to flush its trace buffer (8 times, lines 16505-16522), printing "Increase SCOREP_TOTAL_MEMORY and try again."
- It then reports a hard out-of-memory error:
  - `SCOREP_TOTAL_MEMORY [bytes]: 131063808` (~125 MB, derived from the 128000K set in the sbatch)
  - `Number of pages: 15999`, `Maximum number of pages allocated: 17047`, `Number of pages currently allocated: 17047` — all pages are exhausted
  - `Number of locations: 5` — rank 24 runs on a GPU node with 24 CPUs, creating multiple Score-P thread locations

**Abort signal chain (lines 16588-16608):**
Rank 24 crashes with SIGABRT (signal 6). The backtrace shows:
```
MPI_Testsome (in pipelinedExchange) → SCOREP_EnterRegion → SCOREP_Location_GetCurrentCPULocation → scorep_thread_create_wait_orphan_begin → SCOREP_Tracing_OnLocationCreation → SCOREP_Tracing_GetEventWriter → SCOREP_Memory_HandleOutOfMemory → abort()
```

**Slurm cascade (lines 16609-16617):**
- Line 16609: `srun: error: w23g0006: task 0: Aborted (core dumped)`
- Lines 16611-16615: Slurm detects the abort and terminates all remaining tasks
- Line 16616: `CANCELLED AT 2026-08-08T07:58:37 DUE to SIGNAL Killed` (SIGKILL, signal 9) — this is the source of exit code 9:0

### Code-level Cause
In `collectiveCommunication.cpp:417-432`, the `pipelinedExchange` method spawns a `progress_thread` that calls `MPI_Testsome` in a tight busy-wait loop with `std::this_thread::yield()`:
```cpp
std::thread progress_thread([&]() {
    while (!progress_stop.load(...) && received_count_atomic.load(...) < workgroup_size_) {
        int completed = 0;
        MPI_Testsome(workgroup_size_, input_requests.data(), &completed,
                     completed_indices.data(), completed_statuses.data());
        if (completed != MPI_UNDEFINED && completed > 0) {
            ...
        }
        std::this_thread::yield();
    }
});
```

With `SCOREP_THREAD_MODE=pthread` and `SCOREP_ENABLE_TRACING=true` (set in `run_aix_p2p_full_suite.sbatch`), every instrumented `MPI_Testsome` call generates trace events that fill the 128 MB trace buffer. The GPU controller rank (rank 24) with 24 CPUs generates the most trace locations and events, exhausting memory first.

### Non-deterministic Nature
The previous successful run (job 2807101) also showed 2 trace buffer flushes on rank 24 during perfect_pipelined, but recovered. In the failed run, the giant_pipelined case generated 8 flush attempts before exhausting memory entirely — suggesting system load or scheduling timing on node w23g0006 caused more trace events to accumulate before the flush could complete.

### Remedies
1. **Increase SCOREP_TOTAL_MEMORY** in `run_aix_p2p_full_suite.sbatch` from `128000K` to at least `512000K` or `1G` — this directly addresses the memory exhaustion for pipelined mode.
2. **Disable Score-P tracing** (`SCOREP_ENABLE_TRACING=false`) if tracing data is not needed — the `perfect_pipelined` case already shows trace flushes even on a smaller model, meaning the 128 MB threshold is fundamentally insufficient for pipelined mode with this configuration.
3. **Reduce MPI_Testsome call frequency** in the progress thread in `collectiveCommunication.cpp:423` (e.g., add a small sleep or batch the polling loop) to reduce the volume of instrumented calls.
4. **Set `SCOREP_TRACE_MAX_MAXSIZE` or increase the number of pages** via Score-P configuration variables if available in version 8.4.