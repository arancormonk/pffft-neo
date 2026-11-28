#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Compare the accuracy and the output bits of two fft_accuracy builds on the same inputs.

Both executables (benchmarks/fft_accuracy.c, built against the baseline and
the candidate library) run in parallel shards. A case is a backend, kind,
size, direction and ordering. The comparison fails if

  * a check within one build fails (in place, zreorder, finite output,
    the absolute error limit, the lane count, a missing setup);
  * the builds do not cover the same cases with the same inputs (cases only
    in the candidate are reported, not judged);
  * the candidate is less accurate: its pooled relative RMS error over the
    baseline's exceeds 1 + --case-tol at one case, or the geometric mean of
    that ratio exceeds 1 + --class-tol over the sizes of a backend, kind and
    direction, or over those of one family of sizes in it (powers of two,
    or with a factor 3, 5 or 15), forward and backward alike;
  * with --bits, a case differs in one bit (hash of inputs and outputs).

Accuracy is judged on the ordered transforms: the unordered ones must give
the same values in another layout (a check), so their errors are equal.
"""

import argparse
import concurrent.futures
import csv
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

COLUMNS = ("backend", "lanes", "kind", "n", "direction", "order", "factors", "df", "inputs",
           "rms", "max", "in_hash", "hash", "checks")
KEY = ("backend", "kind", "n", "direction", "order")
MERGED = KEY + ("coverage", "lanes_a", "lanes_b", "factors_a", "factors_b", "rms_a", "rms_b", "ratio",
                "max_a", "max_b", "identical")
BACKENDS = ("native", "w8", "native-df", "w8-df", "native-bf", "w8-bf", "auto")
SCHEDULED = ("native-df", "w8-df", "native-bf", "w8-bf")
# Starting from upstream pffft's calibration (accuracy.py, b58f839) and
# checked on this library: see README.md, "Compare FFT accuracy".
CASE_TOL = 0.05
CLASS_TOL = 0.01
SHOWN = 40


class Error(Exception):
    pass


def run_shard(executable, options, shard, shards):
    command = [str(executable), *options, "--shard", f"{shard}/{shards}"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    # exit status 1 with complete output: a check failed, which the output records
    lines = result.stdout.splitlines()
    end = re.fullmatch(r"# end cases=\d+ failures=(\d+)", lines[-1]) if lines else None
    if not end or result.returncode != (1 if int(end.group(1)) else 0):
        raise Error(f"{executable} --shard {shard}/{shards} failed (exit status {result.returncode}): "
                    f"{result.stderr.strip()}")
    return result.stdout


def parse(texts, label):
    """(meta, rows, errors) of the shards of one build; rows keyed by KEY."""
    meta, rows, errors = {}, {}, []
    for text in texts:
        lines = text.splitlines()
        if not lines or not lines[-1].startswith("# end "):
            raise Error(f"{label}: incomplete fft_accuracy output")
        body = []
        for line in lines:
            if line.startswith("# error:"):
                errors.append(line[len("# error:"):].strip())
            elif line.startswith("#"):
                key, sep, value = line[1:].strip().partition("=")
                if sep and key not in ("end cases",):
                    if meta.setdefault(key, value) != value:
                        raise Error(f"{label}: shards disagree on {key}")
            else:
                body.append(line)
        reader = csv.reader(body)
        if tuple(next(reader, ())) != COLUMNS:
            raise Error(f"{label}: fft_accuracy wrote no CSV of this format")
        for fields in reader:
            if len(fields) != len(COLUMNS):
                raise Error(f"{label}: malformed row {fields}")
            row = dict(zip(COLUMNS, fields))
            row["n"], row["lanes"], row["inputs"] = int(row["n"]), int(row["lanes"]), int(row["inputs"])
            for name in ("rms", "max"):
                row[name] = float(row[name]) if row[name] else None
            key = tuple(row[k] for k in KEY)
            if key in rows:
                raise Error(f"{label}: duplicate case {key}")
            rows[key] = row
    return meta, rows, errors


def family(n):
    return {(False, False): "2^k", (True, False): "3*", (False, True): "5*", (True, True): "15*"}[
        (n % 3 == 0, n % 5 == 0)]


def gmean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else 1.0


def describe(key):
    return " ".join(str(k) for k in key)


def print_differ(differ, a_rows, b_rows, verbose):
    """One line per backend, kind, size and direction that differs, largest error ratio first."""
    groups = {}
    for k, ratio in differ:
        groups.setdefault(k[:4], {})[k[4]] = ratio
    lines = []
    for case, orders in groups.items():
        order = "both" if len(orders) == 2 else next(iter(orders))
        k = case + ("ordered" if "ordered" in orders else "unordered",)
        lines.append((k, order, orders.get("ordered", orders.get("unordered"))))
    if not verbose:
        lines.sort(key=lambda line: -(line[2] or 0))
    count = len(lines)
    if not verbose and count > SHOWN:
        lines = lines[:SHOWN]
        print(f"\ncases that differ (largest B/A first, {SHOWN} of {count}; -v for all):")
    else:
        print("\ncases that differ:")
    print("backend    kind          n direction order     factors A -> B             rms A       rms B      B/A")
    for k, order, ratio in lines:
        a, b = a_rows[k], b_rows[k]
        factors = a["factors"] if a["factors"] == b["factors"] else f"{a['factors'] or '?'} -> {b['factors'] or '?'}"
        if a["lanes"] != b["lanes"]:
            factors += f" ({a['lanes']} -> {b['lanes']} lanes)"
        rms_a = f"{a['rms']:.4e}" if a["rms"] is not None else "-"
        rms_b = f"{b['rms']:.4e}" if b["rms"] is not None else "-"
        shown = f"{ratio:.4f}" if ratio is not None else "-"
        print(f"{k[0]:10} {k[1]:8} {k[2]:8} {k[3]:9} {order:9} {factors:26} {rms_a:>10} {rms_b:>11} {shown:>8}")


def compare(a_meta, a_rows, a_errors, b_meta, b_rows, b_errors, args):
    """Prints the report; returns (failure messages, merged rows)."""
    failed, merged = [], []
    for name in ("simd_arch", "simd_size", "w8", "ldbl_mant_dig", "seeds", "points_per_seed", "hash_only"):
        if a_meta.get(name) != b_meta.get(name):
            print(f"note: the builds differ in {name}: {a_meta.get(name)} / {b_meta.get(name)}")
    if a_meta.get("w8") != b_meta.get("w8"):
        failed.append("the 8-lane backend is available in one build only")
    for label, errors in (("baseline", a_errors), ("candidate", b_errors)):
        for error in errors:
            print(f"{label} error: {error}")
        if errors:
            failed.append(f"{len(errors)} errors in the {label}")

    only_a = sorted(set(a_rows) - set(b_rows))
    only_b = sorted(set(b_rows) - set(a_rows))
    for keys, where in ((only_a, "baseline"), (only_b, "candidate")):
        if keys:
            print(f"{len(keys)} cases only in the {where}, e.g. " + "; ".join(describe(k) for k in keys[:5]))
    # A -df or -bf row exists where the plan allows that schedule, which a new
    # plan may change: noted, unless that backend was asked for. A case the
    # candidate lost elsewhere fails.
    requested = set(args.backends or ())
    lost = [k for k in only_a if k[0] not in SCHEDULED or k[0] in requested]
    if lost:
        failed.append(f"{len(lost)} cases only in the baseline")
    elif only_a:
        print("  (rows of a forced schedule, which the candidate's plans no longer allow at those sizes)")

    common = sorted(set(a_rows) & set(b_rows), key=lambda k: (BACKENDS.index(k[0]) if k[0] in BACKENDS else 99,
                                                              k[1], k[3], k[4], k[2]))
    if not common:
        failed.append("no case in both builds")
    for backend in sorted(requested):
        if not any(k[0] == backend for k in common):
            failed.append(f"no case in both builds for the requested backend {backend}")
    bad_checks = [(label, k, rows[k]["checks"]) for label, rows in (("baseline", a_rows), ("candidate", b_rows))
                  for k in sorted(rows) if rows[k]["checks"]]
    for label, k, checks in bad_checks[:SHOWN]:
        print(f"{label} check failed: {describe(k)}: {checks}")
    if bad_checks:
        failed.append(f"{len(bad_checks)} cases fail a check within a build")

    differ, ratios, case_flags, classes = [], {}, [], {}
    ties = improved = dispatch = 0
    for k in common:
        a, b = a_rows[k], b_rows[k]
        backend, kind, n, direction, order = k
        if a["inputs"] != b["inputs"]:
            failed.append(f"{describe(k)}: different numbers of inputs")
        # unordered backward inputs come from each build's own zreorder()
        if a["in_hash"] != b["in_hash"] and not (order == "unordered" and direction == "backward"):
            failed.append(f"{describe(k)}: different inputs")
        if a["lanes"] != b["lanes"] and backend != "auto":
            failed.append(f"{describe(k)}: {a['lanes']} lanes in the baseline, {b['lanes']} in the candidate")
        identical = a["in_hash"] == b["in_hash"] and a["hash"] == b["hash"]
        dispatch += a["lanes"] != b["lanes"] and backend == "auto"
        ratio = None
        if a["rms"] is not None and b["rms"] is not None:
            if a["rms"] > 0 and b["rms"] > 0:
                ratio = b["rms"] / a["rms"]
            elif a["rms"] == 0 and b["rms"] == 0:
                ratio = 1.0
            elif b["rms"] == 0:
                ratio = 0.0
            else:
                ratio = math.inf
        merged.append(dict(zip(KEY, k), coverage="both", lanes_a=a["lanes"], lanes_b=b["lanes"], factors_a=a["factors"],
                           factors_b=b["factors"], rms_a=a["rms"], rms_b=b["rms"], ratio=ratio, max_a=a["max"],
                           max_b=b["max"], identical=int(identical)))
        if not identical:
            differ.append((k, ratio))
        if order != "ordered" or ratio is None:
            continue
        if ratio > 1 + args.case_tol:
            case_flags.append(k)
        if ratio == 1.0 and a["rms"] == 0:
            ties += 1
        elif ratio == 0.0:
            improved += 1
        elif math.isfinite(ratio):
            ratios.setdefault((backend, kind, direction), []).append(ratio)
            classes.setdefault((backend, kind, direction, family(n)), []).append(ratio)

    print(f"cases in both builds: {len(common)}; bit-identical: {len(common) - len(differ)}; "
          f"differ: {len(differ)}")
    if dispatch:
        print(f"pffft_new_setup() picks other lanes in {dispatch} auto cases")
    same_bits = dict.fromkeys(common, True)
    for k, _ in differ:
        same_bits[k] = False
    hash_only = a_meta.get("hash_only") == "1"
    print()
    print("backend    kind      cases identical  fwd gmean (max)    bwd gmean (max)   identical in all four rows")
    for backend, kind in dict.fromkeys((k[0], k[1]) for k in common):
        keys = [k for k in common if k[:2] == (backend, kind)]
        sizes = sorted({k[2] for k in keys})
        same = sum(1 for k in keys if same_bits[k])
        unchanged = [n for n in sizes if all(same_bits[k] for k in keys if k[2] == n)]
        cells = []
        for direction in ("forward", "backward"):
            values = ratios.get((backend, kind, direction), [])
            cells.append(f"{gmean(values):.4f} ({max(values):.3f})" if values else "-")
        print(f"{backend:10} {kind:8} {len(keys):6} {same:9}  {cells[0]:18} {cells[1]:17} "
              f"{len(unchanged)} of {len(sizes)} sizes")
    if ties or improved:
        print(f"exact (zero error) in both: {ties} cases; zero error in the candidate only: {improved}")

    if not hash_only:
        for cls, values in sorted(ratios.items()):
            mean = gmean(values)
            if mean > 1 + args.class_tol:
                failed.append(f"{' '.join(cls)}: less accurate on average ({mean:.4f} over {len(values)} sizes)")
        for cls, values in sorted(classes.items()):
            mean = gmean(values)
            if mean > 1 + args.class_tol and len(values) < len(ratios[cls[:3]]):
                failed.append(f"{' '.join(cls[:3])} {cls[3]} sizes: less accurate on average "
                              f"({mean:.4f} over {len(values)} sizes)")
        for k in case_flags:
            failed.append(f"{describe(k)}: less accurate ({a_rows[k]['rms']:.4g} -> {b_rows[k]['rms']:.4g})")
    if args.bits and differ:
        failed.append(f"{len(differ)} cases differ in their bits (--bits)")

    if differ:
        print_differ(differ, a_rows, b_rows, args.verbose)
    for keys, rows, coverage, side in ((only_a, a_rows, "baseline_only", "a"), (only_b, b_rows, "candidate_only", "b")):
        for k in keys:
            r = rows[k]
            merged.append(dict(zip(KEY, k), coverage=coverage, **{f"lanes_{side}": r["lanes"],
                                                                 f"factors_{side}": r["factors"],
                                                                 f"rms_{side}": r["rms"], f"max_{side}": r["max"]}))
    return failed, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("baseline", type=Path, help="fft_accuracy built against the baseline library")
    parser.add_argument("candidate", type=Path, help="fft_accuracy built against the candidate library")
    parser.add_argument("--max", type=int, help="largest size (fft_accuracy's default: 65536)")
    parser.add_argument("--sizes", nargs="+", type=int, help="these sizes instead of all up to --max")
    parser.add_argument("--kinds", nargs="+", choices=["real", "complex"])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, help="default: all the builds have")
    parser.add_argument("--seeds", type=int, help="seeds per case, 8192 points each (default 2)")
    parser.add_argument("--seed-base", type=int, help="first seed (default 1)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes")
    parser.add_argument("--bits", action="store_true", help="every case must give the baseline's bits")
    parser.add_argument("--hash-only", action="store_true", help="no reference: compare bits only (fast)")
    parser.add_argument("--case-tol", type=float, default=CASE_TOL,
                        help=f"accepted increase of the error of one case: B/A <= 1 + CASE_TOL (default {CASE_TOL})")
    parser.add_argument("--class-tol", type=float, default=CLASS_TOL,
                        help=f"accepted increase of the geometric mean of B/A over a class (default {CLASS_TOL})")
    parser.add_argument("--output", type=Path, help="new CSV of every case of both builds")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every case that differs")
    args = parser.parse_args()
    if args.jobs < 1 or not all(math.isfinite(t) and t >= 0 for t in (args.case_tol, args.class_tol)):
        parser.error("expected positive --jobs and finite nonnegative tolerances")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing directory")
    options = []
    for name in ("max", "seeds", "seed_base"):
        if getattr(args, name) is not None:
            options += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
    for name in ("sizes", "kinds", "backends"):
        if getattr(args, name):
            options += [f"--{name}", ",".join(str(v) for v in getattr(args, name))]
    if args.hash_only:
        options.append("--hash-only")

    builds = (args.baseline.resolve(), args.candidate.resolve())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [[pool.submit(run_shard, exe, options, shard, args.jobs) for shard in range(args.jobs)]
                   for exe in builds]
        outputs = [[f.result() for f in shards] for shards in futures]
    a_meta, a_rows, a_errors = parse(outputs[0], "baseline")
    b_meta, b_rows, b_errors = parse(outputs[1], "candidate")
    print(f"A = {builds[0]}\nB = {builds[1]}")
    print(f"{a_meta.get('simd_arch')} x{a_meta.get('simd_size')}, 8 lanes {'available' if a_meta.get('w8') == '1' else 'unavailable'}; "
          f"reference: long double of {a_meta.get('ldbl_mant_dig')} bits; seeds {a_meta.get('seeds')}, "
          f"{a_meta.get('points_per_seed')} points per seed and case\n")
    failed, merged = compare(a_meta, a_rows, a_errors, b_meta, b_rows, b_errors, args)
    if args.output:
        with open(args.output, "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MERGED)
            writer.writeheader()
            writer.writerows(merged)
    if failed:
        print("\nFAILED:")
        for message in failed[:SHOWN]:
            print(f"  {message}")
        if len(failed) > SHOWN:
            print(f"  ... and {len(failed) - SHOWN} more")
    else:
        print("\nPASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, Error) as error:
        sys.exit(f"accuracy_compare.py: {error}")
