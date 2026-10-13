#!/usr/bin/env python3
"""Per-workgroup medians for the pure-MPI controller-count sweep."""

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

    print(f"{'file':44s} {'wg':>3s} {'n':>4s} {'assembly_ms':>12s} {'agg_GiB/s':>10s}")
    rows_out = []
    for path in paths:
        per_case = defaultdict(lambda: defaultdict(list))
        with open(path, newline="") as stream:
            reader = csv.DictReader(line for line in stream if not line.startswith("#"))
            for row in reader:
                workgroup = int(float(row["workgroup"]))
                per_case[workgroup]["assembly"].append(float(row["assembly_ms"]))
                per_case[workgroup]["bw"].append(float(row["aggregate_input_gib_s"]))
        tag = Path(path).stem
        for workgroup in sorted(per_case):
            case = per_case[workgroup]
            row = (tag, workgroup, len(case["assembly"]),
                   median(case["assembly"]), median(case["bw"]))
            rows_out.append(row)
            print(f"{row[0]:44s} {row[1]:3d} {row[2]:4d} {row[3]:12.4f} {row[4]:10.3f}")

    if args.output:
        with open(args.output, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "workgroup", "samples", "assembly_median_ms", "aggregate_median_gibs"])
            writer.writerows(rows_out)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
