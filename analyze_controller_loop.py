#!/usr/bin/env python3
"""Reconstruct the AIx P2P controller budget from instrumented timeline CSVs.

Pairs emitted by the instrumented AIx build:
  p2p_poll_begin/end, p2p_collect_begin/end, p2p_select_begin/end,
  p2p_submit_begin/end, p2p_output_flush_begin/end.
Existing events (input_ready, range_inference_*, torch_*, result_send_*) are
used to bound the step and cross-check the receive cadence.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

PAIRS = [
    ("poll", "p2p_poll_begin", "p2p_poll_end"),
    ("collect", "p2p_collect_begin", "p2p_collect_end"),
    ("select", "p2p_select_begin", "p2p_select_end"),
    ("submit", "p2p_submit_begin", "p2p_submit_end"),
    ("flush", "p2p_output_flush_begin", "p2p_output_flush_end"),
]


def load_rank(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def step_budget(rows: list[dict], step: int) -> dict:
    events = [(float(r["time_s"]), r["event"], r) for r in rows if int(r["step"]) == step]
    events.sort()
    if not events:
        return {}
    t0 = events[0][0]
    totals = defaultdict(float)
    counts = defaultdict(int)
    poll_durations = []

    # Prefer per-step budget summary events when present (low-overhead build).
    used_summary = False
    for _, event, row in events:
        if event.startswith("p2p_budget_"):
            block = event[len("p2p_budget_"):]
            totals[block] = float(row["sample_start"]) / 1e3  # us -> ms
            counts[block] = int(row["sample_count"])
            used_summary = True

    if not used_summary:
        for name, begin, end in PAIRS:
            open_ts = None
            for t, event, _ in events:
                if event == begin:
                    open_ts = t
                elif event == end and open_ts is not None:
                    duration = (t - open_ts) * 1e3
                    totals[name] += duration
                    counts[name] += 1
                    if name == "poll":
                        poll_durations.append(duration)
                    open_ts = None

    input_ready = sorted((t - t0) * 1e3 for t, e, _ in events if e == "input_ready")
    span = (events[-1][0] - t0) * 1e3
    busy = sum(totals.values())
    return {
        "step": step,
        "span_ms": span,
        "poll_ms": totals["poll"],
        "collect_ms": totals["collect"],
        "select_ms": totals["select"],
        "submit_ms": totals["submit"],
        "flush_ms": totals["flush"],
        "other_ms": span - busy,
        "poll_count": counts["poll"],
        "poll_median_ms": median(poll_durations) if poll_durations else 0.0,
        "poll_max_ms": max(poll_durations) if poll_durations else 0.0,
        "input_ready_count": len(input_ready),
        "input_last_ready_ms": input_ready[-1] if input_ready else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline_dir", help="directory with aix_p2p_timeline_rank_*.csv")
    parser.add_argument("--rank", type=int, default=0, help="controller world rank")
    parser.add_argument("--skip-first-ml", type=int, default=1)
    args = parser.parse_args()

    path = Path(args.timeline_dir) / f"aix_p2p_timeline_rank_{args.rank}.csv"
    rows = load_rank(path)
    steps = sorted({int(r["step"]) for r in rows})
    budgets = [step_budget(rows, step) for step in steps]
    budgets = [b for b in budgets if b]
    budgets = budgets[args.skip_first_ml:]

    header = (f"{'step':>4} {'span':>8} {'poll':>8} {'collect':>8} {'select':>7} "
              f"{'submit':>8} {'flush':>7} {'other':>8} {'polls':>6} {'p50poll':>8} "
              f"{'p_max':>8} {'last_ready':>10}")
    print(f"controller rank {args.rank}: {path}")
    print(header)
    for b in budgets:
        print(f"{b['step']:>4} {b['span_ms']:>8.2f} {b['poll_ms']:>8.2f} {b['collect_ms']:>8.2f} "
              f"{b['select_ms']:>7.2f} {b['submit_ms']:>8.2f} {b['flush_ms']:>7.2f} "
              f"{b['other_ms']:>8.2f} {b['poll_count']:>6} {b['poll_median_ms']:>8.3f} "
              f"{b['poll_max_ms']:>8.2f} {b['input_last_ready_ms']:>10.2f}")

    def med(key: str) -> float:
        return median(b[key] for b in budgets)

    print()
    print("medians: " + ", ".join(
        f"{key}={med(key):.2f}" for key in
        ("span_ms", "poll_ms", "collect_ms", "select_ms", "submit_ms", "flush_ms",
         "other_ms", "poll_count", "poll_median_ms", "poll_max_ms", "input_last_ready_ms")))


if __name__ == "__main__":
    main()
