#!/usr/bin/env python3
"""Summarize controller-critical-path results from aix_input_assembly_mpi."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_LABELS = {
    "collective_gatherv": "MPI_Gatherv",
    "p2p_credit1": "P2P credit=1",
    "p2p_fullcredits": "P2P credit=23",
}
MODE_ORDER = ["collective_gatherv", "p2p_credit1", "p2p_fullcredits"]
COLORS = {"collective_gatherv": "#0072B2", "p2p_credit1": "#D55E00", "p2p_fullcredits": "#009E73"}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_critical_paths(paths: list[str]) -> dict[tuple[str, int], list[float]]:
    by_case: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="") as stream:
            rows = (line for line in stream if not line.startswith("#"))
            for row in csv.DictReader(rows):
                key = (row["mode"], int(row["scenario_world_ranks"]))
                by_case[key][int(row["iteration"])].append(float(row["assembly_ms"]))

    # The application waits for its slowest controller, so reduce concurrent
    # workgroups to one critical-path measurement per iteration.
    return {key: [max(values) for _, values in sorted(iterations.items())]
            for key, iterations in by_case.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", default="input_assembly")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.input_glob}")
    critical_paths = load_critical_paths(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.prefix}_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "mode", "samples", "critical_path_median_ms", "critical_path_p05_ms", "critical_path_p95_ms"])
        for ranks in (24, 96):
            for mode in MODE_ORDER:
                values = critical_paths.get((mode, ranks), [])
                if values:
                    writer.writerow([f"{ranks}c{ranks // 24}", mode, len(values),
                                     f"{median(values):.6f}", f"{percentile(values, 0.05):.6f}",
                                     f"{percentile(values, 0.95):.6f}"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, layout="constrained")
    for axis, ranks in zip(axes, (24, 96)):
        values = [critical_paths.get((mode, ranks), []) for mode in MODE_ORDER]
        plot_values = [value if value else [float("nan")] for value in values]
        box = axis.boxplot(plot_values, patch_artist=True, showfliers=False)
        for patch, mode in zip(box["boxes"], MODE_ORDER):
            patch.set_facecolor(COLORS[mode])
            patch.set_alpha(0.8)
        axis.set_title(f"{ranks} ranks, {ranks // 24} controller workgroup{'s' if ranks > 24 else ''}")
        axis.set_xticklabels([MODE_LABELS[mode] for mode in MODE_ORDER], rotation=18, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Input assembly critical path (ms)")
    fig.suptitle("Pure MPI Input Assembly: Collective vs AIx-Equivalent P2P")
    figure_path = output_dir / f"{args.prefix}_critical_path.png"
    fig.savefig(figure_path, dpi=300)
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
