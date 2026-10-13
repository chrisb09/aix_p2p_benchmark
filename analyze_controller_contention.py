#!/usr/bin/env python3
"""Analyze staged P2P controller contention sweeps."""

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


STAGES = ["pinned_mpi", "h2d_compute_d2h"]
STAGE_LABELS = {"pinned_mpi": "Pinned MPI input-ready", "h2d_compute_d2h": "Full staged completion"}


def load(paths: list[str]):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="") as stream:
            rows = (line for line in stream if not line.startswith("#"))
            for row in csv.DictReader(rows):
                key = (int(row["scenario_world_ranks"]), row["mpi_thread_requested"],
                       int(row["async_controller"]), int(row["submit_cpu_us"]), row["stage"])
                grouped[key][int(row["iteration"])].append(
                    (float(row["input_ready_ms"]), float(row["stage_complete_ms"])))
    return {key: [(max(value[0] for value in rows), max(value[1] for value in rows))
                  for _, rows in sorted(iterations.items())]
            for key, iterations in grouped.items()}


def label(thread: str, async_controller: int) -> str:
    return f"{thread}, {'async' if async_controller else 'inline'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", default="controller_contention")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.input_glob}")
    data = load(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.prefix}_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "mpi_thread", "async_controller", "submit_cpu_us", "stage", "samples",
                         "input_ready_critical_median_ms", "stage_complete_critical_median_ms"])
        for key in sorted(data):
            ranks, thread, async_controller, cpu_us, stage = key
            values = data[key]
            writer.writerow([f"{ranks}c{ranks // 24}", thread, async_controller, cpu_us, stage, len(values),
                             f"{median(value[0] for value in values):.6f}",
                             f"{median(value[1] for value in values):.6f}"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True)
    for row, ranks in enumerate((24, 96)):
        variants = sorted({(thread, async_controller) for r, thread, async_controller, _, _ in data if r == ranks})
        for column, stage in enumerate(STAGES):
            axis = axes[row, column]
            metric = 0 if stage == "pinned_mpi" else 1
            for thread, async_controller in variants:
                points = []
                for key, values in data.items():
                    r, t, a, cpu_us, s = key
                    if (r, t, a, s) == (ranks, thread, async_controller, stage):
                        points.append((cpu_us, median(value[metric] for value in values)))
                if points:
                    points.sort()
                    axis.plot([point[0] for point in points], [point[1] for point in points], marker="o",
                              label=label(thread, async_controller))
            axis.set_title(f"{ranks}c{ranks // 24}: {STAGE_LABELS[stage]}")
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            if column == 0:
                axis.set_ylabel("Critical-path time (ms)")
            if row == 1:
                axis.set_xlabel("Controller CPU work per submitted range (us)")
    axes[0, 1].legend(loc="upper left")
    figure_path = output_dir / f"{args.prefix}_critical_path.png"
    fig.savefig(figure_path, dpi=300)
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
