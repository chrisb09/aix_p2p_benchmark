#!/usr/bin/env python3
"""Summarize staged input-assembly tests by their per-step slowest controller."""

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


MODE_ORDER = ["collective_gatherv", "p2p_credit1", "p2p_fullcredits"]
STAGE_ORDER = ["pinned_mpi", "h2d", "h2d_compute", "h2d_compute_d2h"]
MODE_LABELS = {"collective_gatherv": "Gatherv", "p2p_credit1": "P2P c=1", "p2p_fullcredits": "P2P c=23"}
STAGE_LABELS = {"pinned_mpi": "Pinned MPI", "h2d": "+H2D", "h2d_compute": "+Compute", "h2d_compute_d2h": "+D2H"}
COLORS = {"collective_gatherv": "#0072B2", "p2p_credit1": "#D55E00", "p2p_fullcredits": "#009E73"}


def load(paths: list[str]) -> dict[tuple[int, int, str, str], list[tuple[float, float]]]:
    grouped: dict[tuple[int, int, str, str], dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="") as stream:
            rows = (line for line in stream if not line.startswith("#"))
            for row in csv.DictReader(rows):
                key = (int(row["scenario_world_ranks"]), int(row.get("async_controller", "0")), row["mode"], row["stage"])
                grouped[key][int(row["iteration"])].append((float(row["input_ready_ms"]), float(row["stage_complete_ms"])))
    return {key: [(max(pair[0] for pair in values), max(pair[1] for pair in values))
                  for _, values in sorted(iterations.items())]
            for key, iterations in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", default="staged_input_assembly")
    args = parser.parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.input_glob}")
    data = load(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = output_dir / f"{args.prefix}_summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "async_controller", "mode", "stage", "samples", "input_ready_median_ms", "stage_complete_median_ms"])
        for ranks in (24, 96):
            for async_controller in (0, 1):
                for stage in STAGE_ORDER:
                    for mode in MODE_ORDER:
                        values = data.get((ranks, async_controller, mode, stage), [])
                        if values:
                            writer.writerow([f"{ranks}c{ranks // 24}", async_controller, mode, stage, len(values),
                                             f"{median(pair[0] for pair in values):.6f}",
                                             f"{median(pair[1] for pair in values):.6f}"])

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.0), layout="constrained", sharey="row")
    for topology_index, ranks in enumerate((24, 96)):
        for async_controller in (0, 1):
            row = topology_index * 2 + async_controller
            for column, metric in enumerate((0, 1)):
                axis = axes[row, column]
                for mode_index, mode in enumerate(MODE_ORDER):
                    x_values = []
                    y_values = []
                    for stage_index, stage in enumerate(STAGE_ORDER):
                        values = data.get((ranks, async_controller, mode, stage), [])
                        if values:
                            x_values.append(stage_index + (mode_index - 1) * 0.12)
                            y_values.append(median(pair[metric] for pair in values))
                    axis.plot(x_values, y_values, marker="o", color=COLORS[mode], label=MODE_LABELS[mode])
                axis.set_xticks(range(len(STAGE_ORDER)), [STAGE_LABELS[stage] for stage in STAGE_ORDER], rotation=20, ha="right")
                axis.grid(axis="y", linestyle="--", alpha=0.35)
                axis.set_title(f"{ranks}c{ranks // 24}, async={async_controller}: {'input ready' if metric == 0 else 'full staged completion'}")
                if column == 0:
                    axis.set_ylabel("Critical-path time (ms)")
    axes[0, 1].legend(loc="upper left")
    figure = output_dir / f"{args.prefix}_critical_path.png"
    fig.savefig(figure, dpi=300)
    print(f"Saved: {summary}")
    print(f"Saved: {figure}")


if __name__ == "__main__":
    main()
