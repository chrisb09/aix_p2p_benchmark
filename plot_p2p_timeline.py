#!/usr/bin/env python3
"""Render clock-corrected AIx P2P event logs as one multi-workgroup timeline plot per ML step."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def get_workgroup_mapping(events):
    """
    Map each workgroup controller world rank to the list of member world ranks.
    Returns: dict { controller_world_rank: [world_ranks in workgroup] }
    """
    controller_ranks = sorted({e["world_rank"] for e in events if e["is_controller"]})
    if not controller_ranks:
        # Fallback if no explicit controller tag
        all_world = sorted({e["world_rank"] for e in events})
        return {0: all_world}

    # Map workgroup_rank -> world_rank per controller
    # We find for each controller, which world_rank corresponds to which workgroup_rank
    wg_members = defaultdict(dict) # controller_rank -> { workgroup_rank: world_rank }

    for e in events:
        w_rank = e["world_rank"]
        wg_rank = e["workgroup_rank"]
        # Find controller world rank for this event's workgroup
        # Controller rank has is_controller == 1 and workgroup_rank == 0
        if e["is_controller"]:
            wg_members[w_rank][0] = w_rank

    # Associate worker ranks to their controller
    for e in events:
        if not e["is_controller"] and e["workgroup_rank"] > 0:
            # Find which controller has this worker
            # Look up peer events or matching timestamps/workgroups
            pass

    # Standard AIxeleratorService RoundRobin grouping:
    # Total world size N, K controllers. Controller c_idx is at controller_ranks[c_idx].
    # Workgroup c_idx contains controller + workers assigned to it.
    all_world_ranks = sorted({e["world_rank"] for e in events})
    num_controllers = len(controller_ranks)

    wg_map = {c: [] for c in controller_ranks}
    for w in all_world_ranks:
        # Check if w is a controller
        if w in controller_ranks:
            wg_map[w].append(w)
        else:
            # Find worker's controller from its events (peer_workgroup_rank == 0)
            worker_events = [e for e in events if e["world_rank"] == w]
            ctrl_for_w = None
            if worker_events:
                # Find controller rank in same group:
                # In AIx RR distribution, controller is rank 0 in workgroup communicator
                # We can match by finding which controller rank received/sent to this worker
                for ctrl in controller_ranks:
                    # Check if controller has events referencing range_first_rank/samples matching worker
                    pass
            # Deterministic RR assignment fallback: w % num_controllers -> controller_ranks[w % num_controllers]
            target_ctrl = controller_ranks[w % num_controllers]
            wg_map[target_ctrl].append(w)

    for c in wg_map:
        wg_map[c] = sorted(list(set(wg_map[c])))
    return wg_map


def synchronize_step_clocks(step_events, wg_map, latency_s: float = 3e-6):
    """
    Per-step clock synchronization aligning worker timestamps to their assigned controller's
    timeline based on P2P causality:
      t_worker_send_start = t_controller_credit_send + latency
    """
    controller_ranks = sorted({e["world_rank"] for e in step_events if e["is_controller"]})
    if not controller_ranks:
        return step_events

    worker_to_ctrl = {}
    for ctrl, workers in wg_map.items():
        for w in workers:
            if w != ctrl:
                worker_to_ctrl[w] = ctrl

    worker_deltas = {}
    for w, ctrl in worker_to_ctrl.items():
        ctrl_events = [e for e in step_events if e["world_rank"] == ctrl]
        w_events = [e for e in step_events if e["world_rank"] == w]
        w_wg_rank = next((e["workgroup_rank"] for e in w_events if e["workgroup_rank"] >= 0), -1)

        c_credit = next((e["time_s"] for e in ctrl_events if e["event"] == "input_credit_send" and e["peer_workgroup_rank"] == w_wg_rank), None)
        w_send = next((e["time_s"] for e in w_events if e["event"] == "input_send_start"), None)

        if c_credit is not None and w_send is not None:
            worker_deltas[w] = (w_send - c_credit) - latency_s
        else:
            c_res = next((e["time_s"] for e in ctrl_events if e["event"] == "result_send_start" and e["peer_workgroup_rank"] == w_wg_rank), None)
            w_res = next((e["time_s"] for e in w_events if e["event"] == "result_received"), None)
            if c_res is not None and w_res is not None:
                worker_deltas[w] = (w_res - c_res) - latency_s

    synchronized_events = []
    for e in step_events:
        e_copy = dict(e)
        w = e_copy["world_rank"]
        if w in worker_deltas:
            e_copy["time_s"] -= worker_deltas[w]
        synchronized_events.append(e_copy)

    return synchronized_events


def render_step(events, step, output_dir: Path, model_name: str):
    step_events = [event for event in events if event["step"] == step]
    if not step_events:
        return

    wg_map = get_workgroup_mapping(step_events)
    step_events = synchronize_step_clocks(step_events, wg_map)

    start = min(event["time_s"] for event in step_events)

    def x_coordinate(time_s):
        return (time_s - start) * 1e3

    max_x = max(x_coordinate(event["time_s"]) for event in step_events)

    controller_ranks = sorted({e["world_rank"] for e in step_events if e["is_controller"]})
    if not controller_ranks:
        controller_ranks = [0]
    min_gpu_rank = min(controller_ranks) if controller_ranks else 0

    num_wgs = len(controller_ranks)
    fig_height = max(4.0, 1.2 + 0.28 * sum(len(wg_map[c]) for c in controller_ranks))

    figure, axes = plt.subplots(
        nrows=num_wgs,
        ncols=1,
        figsize=(16, fig_height),
        sharex=True,
        squeeze=False
    )

    legend_candidates = [
        ("input_send_start", Patch(facecolor="#4c9ed9", label="worker input send -> controller-ready (incl. ack)")),
        ("input_credit_wait_start", Patch(facecolor="#f0c040", label="worker waits for input credit")),
        ("result_send_start", Patch(facecolor="#f4a261", label="controller result-send -> worker output received")),
        ("range_inference_start", Patch(facecolor="#8e6bbd", label="selected ready-range inference")),
        ("controller_input_copy_start", Patch(facecolor="#6c757d", label="controller input-buffer copy")),
        ("controller_output_copy_start", Patch(facecolor="#f6bd60", label="controller output-buffer copy")),
        ("torch_controller_output_copy_start", Patch(facecolor="#d4a373", label="Torch staged output -> controller copy")),
        ("torch_input_stage_alloc_start", Patch(facecolor="#9b5de5", label="Torch input staging allocation")),
        ("torch_input_stage_copy_start", Patch(facecolor="#00bbf9", label="Torch input staging copy")),
        ("torch_output_stage_alloc_start", Patch(facecolor="#f15bb5", label="Torch output staging allocation")),
        ("torch_h2d_start", Patch(facecolor="#2a9d8f", label="H2D transfer")),
        ("torch_forward_start", Patch(facecolor="#151515", label="Torch forward execution")),
        ("torch_d2h_start", Patch(facecolor="#ff0033", label="D2H transfer")),
    ]

    for wg_idx, ctrl_rank in enumerate(controller_ranks):
        axis = axes[wg_idx, 0]
        member_world_ranks = wg_map[ctrl_rank]
        ctrl_events = [e for e in step_events if e["world_rank"] == ctrl_rank]

        # Map world rank to y-axis index in this subplot
        rank_to_y = {r: i for i, r in enumerate(member_world_ranks)}
        ctrl_y = rank_to_y[ctrl_rank]

        for r in member_world_ranks:
            axis.hlines(rank_to_y[r], -0.4, max_x + 0.8, color="0.9", linewidth=0.45, zorder=0)

        # Draw worker lanes
        for r in member_world_ranks:
            if r == ctrl_rank:
                continue
            r_events = [e for e in step_events if e["world_rank"] == r]
            y_pos = rank_to_y[r]

            send = [e["time_s"] for e in r_events if e["event"] == "input_send_start"]
            send_comp = [e["time_s"] for e in r_events if e["event"] == "input_send_complete"]
            res_recv = [e["time_s"] for e in r_events if e["event"] == "result_received"]

            # Workgroup rank of this worker, taken from a non-solver event (solver events store -1).
            worker_wg_rank = next((e["workgroup_rank"] for e in r_events if e["workgroup_rank"] >= 0), -1)

            res_send_start = [e["time_s"] for e in ctrl_events
                              if e["event"] == "result_send_start"
                              and e.get("peer_workgroup_rank") == worker_wg_rank]

            # Credit wait segment: worker blocked on credit receive, before it can send input.
            credit_wait = [e["time_s"] for e in r_events if e["event"] == "input_credit_wait_start"]
            if credit_wait and send and send[0] >= credit_wait[0]:
                axis.broken_barh([(x_coordinate(credit_wait[0]), x_coordinate(send[0]) - x_coordinate(credit_wait[0]))],
                                 (y_pos + 0.31, 0.11), facecolors="#f0c040", edgecolors="none")

            if send and send_comp and send_comp[0] >= send[0]:
                axis.broken_barh([(x_coordinate(send[0]), x_coordinate(send_comp[0]) - x_coordinate(send[0]))],
                                 (y_pos - 0.31, 0.27), facecolors="#4c9ed9", edgecolors="none")

            if res_send_start and res_recv and res_recv[-1] >= res_send_start[0]:
                axis.broken_barh([(x_coordinate(res_send_start[0]), x_coordinate(res_recv[-1]) - x_coordinate(res_send_start[0]))],
                                 (y_pos + 0.04, 0.27), facecolors="#f4a261", edgecolors="none")

            # Per-rank solver entry marker (from the solver timeline): when this rank starts its ML step.
            for t in [e["time_s"] for e in step_events
                      if e["world_rank"] == r and e["event"] == "solver_ml_step_start"]:
                axis.plot(x_coordinate(t), y_pos + 0.46, marker="^", color="#111111",
                          markersize=6, zorder=6, clip_on=False)

            # Controller credit-send tick targeting this worker's workgroup rank.
            for t in [e["time_s"] for e in ctrl_events
                      if e["event"] == "input_credit_send" and e["peer_workgroup_rank"] == worker_wg_rank]:
                axis.plot(x_coordinate(t), y_pos - 0.43, marker="v", color="#f0c040",
                          markersize=6, zorder=6, clip_on=False)

        # Draw controller lane events on controller's y_pos
        for event in ctrl_events:
            if event["event"] == "range_inference_start":
                ends = [c for c in ctrl_events if c["event"] == "range_inference_end"
                        and c["range_first_rank"] == event["range_first_rank"]
                        and c["range_end_rank"] == event["range_end_rank"]
                        and c["time_s"] >= event["time_s"]]
                if ends:
                    begin = x_coordinate(event["time_s"])
                    axis.broken_barh([(begin, x_coordinate(ends[0]["time_s"]) - begin)],
                                     (ctrl_y - 0.24, 0.22), facecolors="#8e6bbd", alpha=0.9)
                    chunk_label = str(event["range_first_rank"]) if event["range_end_rank"] == event["range_first_rank"] + 1 \
                        else f"{event['range_first_rank']}-{event['range_end_rank'] - 1}"
                    axis.text(begin, ctrl_y - 0.48, chunk_label, ha="center", va="top", fontsize=5.5, clip_on=False)

        for start_name, end_name, lane_offset, color in (
            ("controller_input_copy_start", "controller_input_copy_end", -0.40, "#6c757d"),
            ("controller_output_copy_start", "controller_output_copy_end", 0.40, "#f6bd60"),
            ("torch_controller_output_copy_start", "torch_controller_output_copy_end", 0.46, "#d4a373"),
            ("torch_input_stage_alloc_start", "torch_input_stage_alloc_end", 0.52, "#9b5de5"),
            ("torch_input_stage_copy_start", "torch_input_stage_copy_end", 0.60, "#00bbf9"),
            ("torch_output_stage_alloc_start", "torch_output_stage_alloc_end", 0.68, "#f15bb5"),
            ("torch_h2d_start", "torch_h2d_end", 0.04, "#2a9d8f"),
            ("torch_forward_start", "torch_forward_end", 0.16, "#151515"),
            ("torch_d2h_start", "torch_d2h_end", 0.28, "#ff0033"),
        ):
            for event in ctrl_events:
                if event["event"] != start_name:
                    continue
                ends = [c for c in ctrl_events if c["event"] == end_name
                        and c["sample_start"] == event["sample_start"]
                        and c["time_s"] >= event["time_s"]]
                if ends:
                    begin = x_coordinate(event["time_s"])
                    axis.broken_barh([(begin, x_coordinate(ends[0]["time_s"]) - begin)],
                                     (ctrl_y + lane_offset, 0.09), facecolors=color)

                    if start_name == "torch_forward_start":
                        axis.vlines(begin, ctrl_y + 0.14, ctrl_y + 0.27, colors="#00ffcc", linewidth=0.8, zorder=4)

        # Controller solver-entry marker (from the solver timeline).
        for t in [e["time_s"] for e in step_events
                  if e["world_rank"] == ctrl_rank and e["event"] == "solver_ml_step_start"]:
            axis.plot(x_coordinate(t), ctrl_y - 0.43, marker="^", color="#111111",
                      markersize=6, zorder=6, clip_on=False)

        # Global start/end lines
        solver_starts = [e["time_s"] for e in step_events if e["event"] in ("solver_ml_step_start", "ml_step_start")]
        solver_ends = [e["time_s"] for e in step_events if e["event"] in ("solver_ml_step_end", "ml_step_end")]
        g_start = min(solver_starts) if solver_starts else start
        g_end = max(solver_ends) if solver_ends else max(e["time_s"] for e in step_events)
        axis.axvline(x_coordinate(g_start), color="#2a9d8f", linestyle="--", linewidth=0.8)
        axis.axvline(x_coordinate(g_end), color="#e76f51", linestyle="--", linewidth=0.8)

        # Y-axis formatting: show world_rank and color code Local vs Remote
        y_ticks = list(range(len(member_world_ranks)))
        y_labels = []
        tick_colors = []
        for r in member_world_ranks:
            if r == ctrl_rank:
                label = f"r{r} [GPU Ctrl]"
                color = "#d62728" # Red/Gold for Controller
            elif r >= min_gpu_rank:
                label = f"r{r} [Local (c23g)]"
                color = "#2ca02c" # Green for Local Intra-Node
            else:
                label = f"r{r} [Remote (c23mm)]"
                color = "#1f77b4" # Blue for Remote Inter-Node
            y_labels.append(label)
            tick_colors.append(color)

        axis.set_yticks(y_ticks)
        axis.set_yticklabels(y_labels, fontsize=6.5)
        for tick_label, col in zip(axis.get_yticklabels(), tick_colors):
            tick_label.set_color(col)
            if "Ctrl" in tick_label.get_text():
                tick_label.set_weight("bold")

        axis.set_ylabel(f"GPU Workgroup {wg_idx}\n(Controller r{ctrl_rank})", fontsize=7.5, weight="bold")
        axis.set_ylim(-0.8, len(member_world_ranks) - 0.2)
        axis.set_xlim(-0.4, max_x + 0.8)
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="x", alpha=0.15, linewidth=0.4)

    axes[-1, 0].set_xlabel("Clock-corrected time since first event (ms)", fontsize=8)

    # Controller performance lines
    controller_lines = []
    for ctrl_rank in controller_ranks:
        c_events = [e for e in step_events if e["world_rank"] == ctrl_rank]
        f_starts = sorted(e["time_s"] for e in c_events if e["event"] == "torch_forward_start")
        f_ends = sorted(e["time_s"] for e in c_events if e["event"] == "torch_forward_end")
        f_sum_ms = sum((b - a) * 1e3 for a, b in zip(f_starts, f_ends))
        c_ends = [e["time_s"] for e in c_events if e["event"] == "solver_ml_step_end"]
        c_end = max(c_ends) if c_ends else max(e["time_s"] for e in step_events)
        first_f_ms = (f_starts[0] - start) * 1e3 if f_starts else float("nan")
        c_end_ms = (c_end - start) * 1e3
        controller_lines.append(
            f"ctrl r{ctrl_rank}: first fwd +{first_f_ms:.2f}ms; fwd sum {f_sum_ms:.2f}ms; end +{c_end_ms:.2f}ms"
        )

    present_events = {e["event"] for e in step_events}
    legend_handles = [p for name, p in legend_candidates if name in present_events]

    if "solver_ml_step_start" in present_events:
        legend_handles.append(Line2D(
            [], [], color="#111111", marker="^", linestyle="None", markersize=6,
            label="solver ML step start (this rank enters AIx)"))
    if "input_credit_send" in present_events:
        legend_handles.append(Line2D(
            [], [], color="#f0c040", marker="v", linestyle="None", markersize=6,
            label="controller sends input credit to this worker"))

    # Add Tier badges to legend
    legend_handles.extend([
        Patch(facecolor="#2ca02c", label="Local Worker (c23g Intra-Node)"),
        Patch(facecolor="#1f77b4", label="Remote Worker (c23mm Inter-Node)"),
    ])

    figure.suptitle(f"AIx P2P Hetjob ML Step {step} | Model: {model_name} | {num_wgs} GPU Workgroups", fontsize=11, y=0.99)
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                  fontsize=6.0, frameon=False, ncol=min(4, max(1, len(legend_handles))))

    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_dir / f"p2p_timeline_step_{step:03d}.png", dpi=180)
    plt.close(figure)


def write_summary(events, output_dir: Path, selected_steps, model_name: str):
    rows = []
    for step in selected_steps:
        step_events = [event for event in events if event["step"] == step]
        if not step_events:
            continue
        wg_map = get_workgroup_mapping(step_events)
        step_events = synchronize_step_clocks(step_events, wg_map)
        starts = [e["time_s"] for e in step_events if e["event"] in ("solver_ml_step_start", "ml_step_start")]
        ends = [e["time_s"] for e in step_events if e["event"] in ("solver_ml_step_end", "ml_step_end")]
        ready = [e["time_s"] for e in step_events if e["event"] == "input_ready"]
        range_calls = sum(event["event"] == "range_inference_start" for event in step_events)
        forward_starts = sorted(e["time_s"] for e in step_events if e["event"] == "torch_forward_start")
        forward_ends = sorted(e["time_s"] for e in step_events if e["event"] == "torch_forward_end")
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
