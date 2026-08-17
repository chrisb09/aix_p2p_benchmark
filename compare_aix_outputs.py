#!/usr/bin/env python3
"""Compare raw binary outputs between collective and P2P AIx benchmark runs."""

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np


def load_manifest(path: Path):
    with path.open() as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_json", type=Path, help="Path to reference (usually collective) output manifest JSON")
    parser.add_argument("candidate_json", type=Path, help="Path to candidate (usually P2P) output manifest JSON")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--require-bitwise", action="store_true", help="Fail if any bitwise mismatch is found")
    parser.add_argument("--report", type=Path, help="Optional output path for detailed JSON comparison report")
    args = parser.parse_args()

    ref_meta = load_manifest(args.reference_json)
    cand_meta = load_manifest(args.candidate_json)

    for key in ("format", "dtype", "shape", "input_seed"):
        if ref_meta.get(key) != cand_meta.get(key):
            sys.exit(f"Metadata mismatch on key '{key}': reference={ref_meta.get(key)} vs candidate={cand_meta.get(key)}")

    ref_bin = args.reference_json.with_suffix(".f32")
    cand_bin = args.candidate_json.with_suffix(".f32")

    if not ref_bin.exists():
        sys.exit(f"Reference binary data not found: {ref_bin}")
    if not cand_bin.exists():
        sys.exit(f"Candidate binary data not found: {cand_bin}")

    shape = tuple(ref_meta["shape"])
    total_elements = math.prod(shape)

    ref_data = np.memmap(ref_bin, dtype="<f4", mode="r", shape=shape)
    cand_data = np.memmap(cand_bin, dtype="<f4", mode="r", shape=shape)

    bitwise_mismatches = 0
    first_mismatch_idx = None
    non_finite_mismatch = 0
    tolerance_failures = 0
    max_abs_err = 0.0
    sum_abs_err = 0.0
    sum_sq_err = 0.0
    max_rel_err = 0.0

    num_steps = shape[0]
    for step in range(num_steps):
        r_step = ref_data[step]
        c_step = cand_data[step]

        r_uint = r_step.view(np.uint32)
        c_uint = c_step.view(np.uint32)

        if np.array_equal(r_uint, c_uint):
            continue

        diff_mask = (r_uint != c_uint)
        mismatches_step = int(np.sum(diff_mask))
        bitwise_mismatches += mismatches_step

        if first_mismatch_idx is None:
            flat_diff = np.where(r_uint.ravel() != c_uint.ravel())[0]
            if len(flat_diff) > 0:
                first_idx = np.unravel_index(flat_diff[0], r_step.shape)
                first_mismatch_idx = [step] + [int(x) for x in first_idx]

        nf_r = ~np.isfinite(r_step)
        nf_c = ~np.isfinite(c_step)
        non_finite_mismatch += int(np.sum(nf_r != nf_c))

        r_diff = r_step[diff_mask].astype(np.float64)
        c_diff = c_step[diff_mask].astype(np.float64)
        abs_diff = np.abs(r_diff - c_diff)

        tol_thresh = args.atol + args.rtol * np.abs(r_diff)
        tolerance_failures += int(np.sum(abs_diff > tol_thresh))

        step_max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        max_abs_err = max(max_abs_err, step_max_abs)
        sum_abs_err += float(np.sum(abs_diff))
        sum_sq_err += float(np.sum(np.square(abs_diff)))

        nonzero = np.abs(r_diff) > 1e-12
        if np.any(nonzero):
            rel_diff = abs_diff[nonzero] / np.abs(r_diff[nonzero])
            max_rel_err = max(max_rel_err, float(np.max(rel_diff)))

    exact_match = (bitwise_mismatches == 0)
    mean_abs_err = float(sum_abs_err / total_elements) if total_elements > 0 else 0.0
    rms_err = float(math.sqrt(sum_sq_err / total_elements)) if total_elements > 0 else 0.0

    report_data = {
        "reference_model": ref_meta.get("model", "unknown"),
        "candidate_model": cand_meta.get("model", "unknown"),
        "shape": list(shape),
        "total_elements": total_elements,
        "exact_bitwise_match": exact_match,
        "bitwise_mismatch_count": bitwise_mismatches,
        "bitwise_mismatch_fraction": float(bitwise_mismatches / total_elements) if total_elements > 0 else 0.0,
        "first_bitwise_mismatch_index": first_mismatch_idx,
        "non_finite_mismatch_count": non_finite_mismatch,
        "tolerance_failure_count": tolerance_failures,
        "tolerance_failure_fraction": float(tolerance_failures / total_elements) if total_elements > 0 else 0.0,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_absolute_error": max_abs_err,
        "mean_absolute_error": mean_abs_err,
        "rms_error": rms_err,
        "max_relative_error": max_rel_err,
        "passed": (tolerance_failures == 0) and (not args.require_bitwise or exact_match)
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as stream:
            json.dump(report_data, stream, indent=2)

    print(f"=== AIx Output Comparison: {ref_meta.get('model')} ===")
    print(f"Total elements: {total_elements}")
    print(f"Exact bitwise match: {exact_match} (mismatches: {bitwise_mismatches})")
    if first_mismatch_idx:
        print(f"First bitwise mismatch at index: {first_mismatch_idx}")
    print(f"Tolerance check (atol={args.atol}, rtol={args.rtol}): {'PASSED' if tolerance_failures == 0 else 'FAILED'}")
    print(f"  Tolerance failures: {tolerance_failures}")
    print(f"  Max absolute error: {max_abs_err:.6e}")
    print(f"  Max relative error: {max_rel_err:.6e}")
    print(f"  RMS error:          {rms_err:.6e}")

    if not report_data["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
