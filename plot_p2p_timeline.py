#!/usr/bin/env python3
"""Render clock-corrected AIx P2P event logs as one timeline per ML step."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_events(directory: Path):
    events = []
    paths = sorted(directory.glob("aix_p2p_timeline_rank_*.csv"))
    paths += sorted(directory.glob("aix_p2p_solver_timeline_rank_*.csv"))
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                for key in (
                    "step",
                    "world_rank",
                    "workgroup_rank",
                    "is_controller",
                    "peer_workgroup_rank",
                    "range_first_rank",
                    "range_end_rank",
                    "sample_start",
                    "sample_count",
                ):
                    row[key] = int(row[key])
                row["time_s"] = float(row["time_s"])
                events.append(row)
    return events


def event_times(events, name, peer=None, rank=None):
    return [event["time_s"] for event in events if event["event"] == name
            and (peer is None or event["peer_workgroup_rank"] == peer)
            and (rank is None or event["workgroup_rank"] == rank)]


def render_step(events, step, output_dir: Path, model_name: str):
    step_events = [event for event in events if event["step"] == step]
    if not step_events:
        return

    ranks = sorted({event["workgroup_rank"] for event in step_events if event["workgroup_rank"] >= 0})
    start = min(event["time_s"] for event in step_events)

    def x_coordinate(time_s):
        return (time_s - start) * 1e3

    max_x = max(x_coordinate(event["time_s"]) for event in step_events)
    figure, axis = plt.subplots(figsize=(16, max(3.0, 1.0 + 0.16 * len(ranks))))
    for rank in ranks:
        axis.hlines(rank, -0.4, max_x + 0.8, color="0.9", linewidth=0.45, zorder=0)

    # Each worker lane has two unambiguous durations without cluttering it with cross-rank lines.
    for rank in ranks:
        if rank == 0:
            continue
        send = event_times(step_events, "input_send_start", 0, rank)
        send_complete = event_times(step_events, "input_send_complete", 0, rank)
        received = event_times(step_events, "result_received", 0, rank)
        if send and send_complete and send_complete[0] >= send[0]:
            axis.broken_barh([(x_coordinate(send[0]), x_coordinate(send_complete[0]) - x_coordinate(send[0]))],
                             (rank - 0.31, 0.27), facecolors="#4c9ed9", edgecolors="none")
        if send_complete and received and received[-1] >= send_complete[0]:
            axis.broken_barh([(x_coordinate(send_complete[0]), x_coordinate(received[-1]) - x_coordinate(send_complete[0]))],
                             (rank + 0.04, 0.27), facecolors="#f4a261", edgecolors="none")

    forward_lane = -0.24
    range_label_index = 0
    for event in step_events:
        if event["event"] != "range_inference_start":
            continue
        ends = [candidate for candidate in step_events if candidate["event"] == "range_inference_end"
                and candidate["range_first_rank"] == event["range_first_rank"]
                and candidate["range_end_rank"] == event["range_end_rank"]
                and candidate["time_s"] >= event["time_s"]]
        if ends:
            begin = x_coordinate(event["time_s"])
            axis.broken_barh([(begin, x_coordinate(ends[0]["time_s"]) - begin)],
                             (forward_lane, 0.22), facecolors="#8e6bbd", alpha=0.9)
            chunk_label = str(event["range_first_rank"]) if event["range_end_rank"] == event["range_first_rank"] + 1 \
                else f"{event['range_first_rank']}-{event['range_end_rank'] - 1}"
            axis.text(begin, -0.48 - 0.15 * (range_label_index % 2), chunk_label,
                      ha="center", va="top", fontsize=5.5, clip_on=False)
            range_label_index += 1

    for start_name, end_name, lane, color in (
        ("controller_input_copy_start", "controller_input_copy_end", -0.40, "#6c757d"),
        ("controller_output_copy_start", "controller_output_copy_end", 0.40, "#f6bd60"),
        ("torch_input_stage_alloc_start", "torch_input_stage_alloc_end", 0.52, "#9b5de5"),
        ("torch_input_stage_copy_start", "torch_input_stage_copy_end", 0.60, "#00bbf9"),
        ("torch_output_stage_alloc_start", "torch_output_stage_alloc_end", 0.68, "#f15bb5"),
        ("torch_h2d_start", "torch_h2d_end", 0.04, "#2a9d8f"),
        ("torch_forward_start", "torch_forward_end", 0.16, "#151515"),
        ("torch_d2h_start", "torch_d2h_end", 0.28, "#e76f51"),
    ):
        for event in step_events:
            if event["event"] != start_name:
                continue
            ends = [candidate for candidate in step_events if candidate["event"] == end_name
                    and candidate["sample_start"] == event["sample_start"]
                    and candidate["time_s"] >= event["time_s"]]
            if ends:
                begin = x_coordinate(event["time_s"])
                axis.broken_barh([(begin, x_coordinate(ends[0]["time_s"]) - begin)],
                                 (lane, 0.09), facecolors=color)

    solver_starts = event_times(step_events, "solver_ml_step_start")
    solver_ends = event_times(step_events, "solver_ml_step_end")
    global_start = min(solver_starts) if solver_starts else min(event["time_s"] for event in step_events)
    global_end = max(solver_ends) if solver_ends else max(event["time_s"] for event in step_events)
    axis.axvline(x_coordinate(global_start), color="#2a9d8f", linestyle="--", linewidth=0.8)
    axis.axvline(x_coordinate(global_end), color="#e76f51", linestyle="--", linewidth=0.8)

    controller_lines = []
    controller_world_ranks = sorted({event["world_rank"] for event in step_events if event["is_controller"]})
    for controller_rank in controller_world_ranks:
        controller_events = [event for event in step_events if event["world_rank"] == controller_rank]
        forward_starts = sorted(event["time_s"] for event in controller_events if event["event"] == "torch_forward_start")
        forward_ends = sorted(event["time_s"] for event in controller_events if event["event"] == "torch_forward_end")
        forward_sum_ms = sum((end - begin) * 1e3 for begin, end in zip(forward_starts, forward_ends))
        controller_solver_starts = [event["time_s"] for event in step_events
                                    if event["world_rank"] == controller_rank and event["event"] == "solver_ml_step_start"]
        controller_solver_ends = [event["time_s"] for event in step_events
                                  if event["world_rank"] == controller_rank and event["event"] == "solver_ml_step_end"]
        controller_end = max(controller_solver_ends) if controller_solver_ends else global_end
        first_forward_ms = (forward_starts[0] - global_start) * 1e3 if forward_starts else float("nan")
        controller_end_ms = (controller_end - global_start) * 1e3
        controller_lines.append(
            f"controller world {controller_rank}: first forward +{first_forward_ms:.3f} ms; "
            f"forward sum {forward_sum_ms:.3f} ms; controller end +{controller_end_ms:.3f} ms")

    axis.set_title(f"AIx P2P ML step {step} | model: {model_name}", fontsize=10)
    axis.set_xlabel("Clock-corrected time since first event (ms)", fontsize=8)
    axis.set_ylabel("Workgroup rank", fontsize=8)
    axis.set_yticks(ranks)
    axis.set_yticklabels([f"r{rank}" for rank in ranks], fontsize=6)
    axis.set_ylim(-1.15, max(ranks, default=0) + 0.45)
    axis.set_xlim(-0.4, max_x + 0.8)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="x", alpha=0.15, linewidth=0.4)
    axis.legend(handles=[
        Patch(facecolor="#4c9ed9", label="worker blocking input send"),
        Patch(facecolor="#f4a261", label="worker send complete -> output received"),
        Patch(facecolor="#8e6bbd", label="selected ready-range inference"),
        Patch(facecolor="#6c757d", label="controller input-buffer copy"),
        Patch(facecolor="#f6bd60", label="controller output-buffer copy"),
        Patch(facecolor="#9b5de5", label="Torch input staging allocation"),
        Patch(facecolor="#00bbf9", label="Torch input staging copy"),
        Patch(facecolor="#f15bb5", label="Torch output staging allocation"),
        Patch(facecolor="#2a9d8f", label="H2D"),
        Patch(facecolor="#151515", label="Torch forward"),
        Patch(facecolor="#e76f51", label="D2H"),
    ], bbox_to_anchor=(0.0, 1.01), loc="lower left", fontsize=6, frameon=False, ncol=6)
    axis.text(0.005, 0.97,
              f"global ML start/end: dashed green/red; global span {(global_end - global_start) * 1e3:.3f} ms\n" +
              "\n".join(controller_lines),
              transform=axis.transAxes, ha="left", va="top", fontsize=6,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82})
    figure.tight_layout()
    figure.savefig(output_dir / f"p2p_timeline_step_{step:03d}.png", dpi=180)
    plt.close(figure)
    print(f"step {step}: global ML start {global_start:.9f}s; global ML span {(global_end - global_start) * 1e3:.3f} ms")
    for line in controller_lines:
        print(f"  {line}")


def write_summary(events, output_dir: Path, selected_steps, model_name: str):
    rows = []
    for step in selected_steps:
        step_events = [event for event in events if event["step"] == step]
        starts = event_times(step_events, "solver_ml_step_start") or event_times(step_events, "ml_step_start")
        ends = event_times(step_events, "solver_ml_step_end") or event_times(step_events, "ml_step_end")
        ready = event_times(step_events, "input_ready")
        range_calls = sum(event["event"] == "range_inference_start" for event in step_events)
        forward_starts = sorted(event_times(step_events, "torch_forward_start"))
        forward_ends = sorted(event_times(step_events, "torch_forward_end"))
        forwards = len(forward_starts)
        forward_sum_ms = sum((end - begin) * 1e3 for begin, end in zip(forward_starts, forward_ends))
        rows.append({
            "step": step,
            "model": model_name,
            "first_start_s": min(starts) if starts else "",
            "last_end_s": max(ends) if ends else "",
            "step_span_ms": (max(ends) - min(starts)) * 1e3 if starts and ends else "",
            "first_input_ready_s": min(ready) if ready else "",
            "input_ready_span_ms": (max(ready) - min(ready)) * 1e3 if ready else "",
            "range_inference_calls": range_calls,
            "torch_forward_calls": forwards,
            "torch_forward_sum_ms": forward_sum_ms,
        })
    with (output_dir / "p2p_timeline_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline_dir", type=Path)
    parser.add_argument("--step", type=int, action="append", help="Render only this step; repeat to select several.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    events = read_events(args.timeline_dir)
    if not events:
        raise SystemExit("No aix_p2p_timeline_rank_*.csv files found.")
    output_dir = args.output_dir or args.timeline_dir / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.timeline_dir / "timeline_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    model_name = metadata.get("model", "unknown")
    warmup_steps = metadata.get("warmup_steps", 0)
    selected_steps = args.step or [step for step in sorted({event["step"] for event in events}) if step >= warmup_steps]
    for step in selected_steps:
        render_step(events, step, output_dir, model_name)
    write_summary(events, output_dir, selected_steps, model_name)


if __name__ == "__main__":
    main()
