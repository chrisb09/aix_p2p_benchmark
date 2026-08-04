# AIx P2P Benchmark

This direct-MPI fixture isolates AIxeleratorService communication and controller scheduling from CMI and terrain-solver preprocessing.

It mirrors the packed `perfect_cuda.pt` mini-app model contract:

- input: `float[N, 18]`;
- output: `float[N]`;
- default local sample count: `129600` (`1920 * 1080 / 16`); and
- default AIx batch size: `4000000`, which covers the complete 25-rank heterogeneous workload in one controller batch when all inputs are ready.

## Build

```bash
cmake -S . -B build -DAIXELERATOR_INSTALL_PREFIX=/path/to/AIxeleratorService/INSTALL
cmake --build build --parallel
```

Build with `-DWITH_SCOREP=ON` and configure with `CXX=scorep-mpicxx` for Score-P user regions.

## Run

```bash
mpirun -np 4 ./build/aix_p2p_benchmark \
  --model ../mini_app/train_models/model_a/perfect_cuda.pt
```

`AIX_COMMUNICATION_MODE=collective` is the default. The pipelined AIx branch enables its point-to-point controller loop with:

```bash
export AIX_COMMUNICATION_MODE=pipelined
```

The root rank writes one CSV row per rank and measured step. `global_step_ms` is the maximum rank-local duration and represents the synchronous coupling-step completion time.

`run_aix_p2p.sbatch` builds and launches the fixture on a four-GPU node. Select the baseline or P2P implementation through `AIXELERATOR_INSTALL_PREFIX`.

`run_aix_p2p_het.sbatch` creates a heterogeneous allocation with 24 one-core solver ranks on `c23mm` and one 24-core GPU-controller/solver rank plus one GPU on `c23g`. This is 25 solver ranks in total; the 24 CPU-node ranks communicate with the remote controller.

The launchers default to one warm-up call and 10 measured calls. P2P runs additionally write corrected-clock CSV events to `aix_p2p_timeline_*`. Render all calls, or one selected measured call, with:

```bash
python3 plot_p2p_timeline.py aix_p2p_timeline_het_pipelined_<job-id>
python3 plot_p2p_timeline.py aix_p2p_timeline_het_pipelined_<job-id> --step 10
```

The renderer creates one PNG per step and `p2p_timeline_summary.csv`, including range-inference and actual Torch-forward call counts.

Use `render_p2p_timeline.sbatch` to render completed timelines on a 12-core `devel` allocation instead of the login node.

For a short Score-P trace, pass `WITH_SCOREP=ON`, a Score-P-enabled AIx install, and set `SCOREP_ENABLE_TRACING=true`, `SCOREP_ENABLE_PROFILING=false`, and a small `AIX_P2P_MEASURED_STEPS` value. The batch script writes the experiment directory as `scorep_aix_<mode>_<job-id>` by default.
