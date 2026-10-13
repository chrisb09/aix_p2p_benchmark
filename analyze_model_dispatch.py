#!/usr/bin/env python3
"""Compare mock-kernel vs real-TorchScript per-range dispatch costs."""

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
    "collective_gatherv": "Gatherv",
    "p2p_credit1": "P2P c=1",
    "p2p_fullcredits": "P2P c=23",
}
STAGE_LABELS = {
    "pinned_mpi": "Pinned MPI",
    "h2d": "H2D",
    "h2d_compute": "H2D+Compute",
    "h2d_compute_d2h": "H2D+Compute+D2H",
}
STAGE_ORDER = ["pinned_mpi", "h2d", "h2d_compute", "h2d_compute_d2h"]
MODE_ORDER = ["collective_gatherv", "p2p_credit1", "p2p_fullcredits"]
MODEL_COLORS = {"mock": "#0072B2", "watercnn": "#D55E00"}


def load(paths: list[str]):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="") as stream:
            rows = (line for line in stream if not line.startswith("#"))
            for row in csv.DictReader(rows):
                key = (row["model"], int(row["async_controller"]), row["mode"], row["stage"])
                grouped[key][(int(row["workgroup"]), int(row["iteration"]))].append(row)
    reduced = {}
    for key, cells in grouped.items():
        per_iteration = defaultdict(list)
        for (_, iteration), rows in cells.items():
            worst_ready = max(float(row["input_ready_ms"]) for row in rows)
            worst_complete = max(float(row["stage_complete_ms"]) for row in rows)
            submit_cpu = sum(float(row.get("submit_cpu_sum_ms") or 0.0) for row in rows)
            scheduled = sum(int(float(row.get("scheduled_ranges") or 0)) for row in rows)
            per_iteration[iteration].append((worst_ready, worst_complete, submit_cpu, scheduled))
        reduced[key] = per_iteration
    return reduced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", default="model_dispatch")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.input_glob}")
    grouped = load(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for key in sorted(grouped):
        model, async_controller, mode, stage = key
        per_iteration = grouped[key]
        n = len(per_iteration)
        values = list(per_iteration.values())
        ready = median(v[0][0] for v in values)
        complete = median(v[0][1] for v in values)
        submit_cpu = median(v[0][2] for v in values)
        ranges = median(v[0][3] for v in values)
        per_range = submit_cpu / ranges if ranges else float("nan")
        summary_rows.append({
            "model": model,
            "async_controller": async_controller,
            "mode": mode,
            "stage": stage,
            "samples": n,
            "input_ready_median_ms": ready,
            "stage_complete_median_ms": complete,
            "submit_cpu_sum_median_ms": submit_cpu,
            "scheduled_ranges_median": ranges,
            "submit_cpu_per_range_us": per_range * 1000.0,
        })

    summary_path = output_dir / f"{args.prefix}_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            for field, value in row.items():
                if isinstance(value, float):
                    row[field] = f"{value:.6f}"
            writer.writerow(row)

    models = sorted({row["model"] for row in summary_rows})
    for async_controller in sorted({row["async_controller"] for row in summary_rows}):
        stages_present = [stage for stage in STAGE_ORDER
                          if any(row["stage"] == stage for row in summary_rows)]
        figure, axes = plt.subplots(1, len(stages_present), figsize=(4.2 * len(stages_present), 4.2),
                                    sharey=False, layout="constrained")
        if len(stages_present) == 1:
            axes = [axes]
        for axis, stage in zip(axes, stages_present):
            width = 0.36
            modes_present = [mode for mode in MODE_ORDER
                             if any(row["mode"] == mode and row["stage"] == stage for row in summary_rows)]
            for model_index, model in enumerate(models):
                positions = []
                ready_values = []
                for mode_index, mode in enumerate(modes_present):
                    match = next((row for row in summary_rows
                                  if row["model"] == model and row["mode"] == mode
                                  and row["stage"] == stage and row["async_controller"] == async_controller), None)
                    if match:
                        positions.append(mode_index + (model_index - 0.5) * width)
                        ready_values.append(match["input_ready_median_ms"])
                axis.bar(positions, ready_values, width=width, color=MODEL_COLORS.get(model, "#999999"),
                         label=model)
            axis.set_xticks(range(len(modes_present)), [MODE_LABELS.get(mode, mode) for mode in modes_present])
            axis.set_title(STAGE_LABELS.get(stage, stage))
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            if axis is axes[0]:
                axis.set_ylabel("Worst-controller input-ready (ms)")
        axes[-1].legend(loc="upper right")
        figure.suptitle(f"Mock vs real WaterCNN dispatch (async={async_controller})")
        figure_path = output_dir / f"{args.prefix}_async{async_controller}.png"
        figure.savefig(figure_path, dpi=300)
        print(f"Saved: {figure_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
