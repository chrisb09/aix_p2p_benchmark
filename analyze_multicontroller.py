#!/usr/bin/env python3
"""Per-workgroup medians for the multi-controller interference test."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No input CSVs matched: {args.input_glob}")

    rows_out = []
    for path in paths:
        per_case = defaultdict(lambda: defaultdict(list))
        with open(path, newline="") as stream:
            reader = csv.DictReader(line for line in stream if not line.startswith("#"))
            for row in reader:
                workgroup = int(float(row["workgroup"]))
                per_case[workgroup]["ready"].append(float(row["input_ready_ms"]))
                per_case[workgroup]["complete"].append(float(row["stage_complete_ms"]))
                per_case[workgroup]["submit"].append(float(row["submit_cpu_sum_ms"]))
        tag = Path(path).stem
        for workgroup in sorted(per_case):
            case = per_case[workgroup]
            rows_out.append((tag, workgroup, len(case["ready"]),
                             median(case["ready"]), median(case["complete"]), median(case["submit"])))

    header = f"{'file':48s} {'wg':>3s} {'n':>4s} {'ready_ms':>10s} {'complete_ms':>12s} {'submit_cpu_ms':>14s}"
    print(header)
    for tag, workgroup, n, ready, complete, submit in rows_out:
        print(f"{tag:48s} {workgroup:3d} {n:4d} {ready:10.3f} {complete:12.3f} {submit:14.4f}")

    if args.output:
        with open(args.output, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "workgroup", "samples", "input_ready_median_ms",
                             "stage_complete_median_ms", "submit_cpu_median_ms"])
            for row in rows_out:
                writer.writerow(row)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
