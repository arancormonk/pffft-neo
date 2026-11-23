#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Time two FFT benchmark executables on the same host and compare per-case medians."""

import argparse
import csv
import gzip
import itertools
import math
from pathlib import Path
from statistics import median
import subprocess
import sys

from wide_size_table import AUXILIARY, GATED

COLUMNS = ("build", "precision", "backend", "kind", "operation", "n", "repeat",
           "iterations", "seconds", "ns_per_call", "checksum")
OPERATIONS = GATED + AUXILIARY


def sample(executable, label, kind, operation, size, seconds, repeat):
    # A fresh process gives each build the same allocation and warmup sequence.
    result = subprocess.run([str(executable), kind, operation, str(size), "1", str(seconds)],
                            check=True, stdout=subprocess.PIPE, text=True)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(COLUMNS) - 1:
        raise ValueError(f"{executable}: unexpected benchmark output")
    row = dict(zip(COLUMNS, [label] + rows[0]))
    timing = float(row["ns_per_call"])
    if ((row["precision"], row["kind"], row["operation"], int(row["n"])) != ("float", kind, operation, size)
            or not row["backend"] or not math.isfinite(timing) or timing <= 0):
        raise ValueError(f"{executable}: invalid measurement {row}")
    row["repeat"] = repeat
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new raw sample CSV or CSV.gz")
    parser.add_argument("--sizes", nargs="+", type=int, default=[96, 320, 1024])
    parser.add_argument("--kinds", nargs="+", choices=["real", "complex"], default=["real", "complex"])
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS + ["all"], default=["forward", "pipeline"])
    parser.add_argument("--samples", type=int, default=5, help="paired measurements per case (default 5)")
    parser.add_argument("--seconds", type=float, default=0.05, help="target time per measurement (default 0.05)")
    parser.add_argument("--labels", nargs=2, default=["baseline", "candidate"], metavar=("BASE", "CAND"))
    args = parser.parse_args()
    if not 0 < args.seconds <= 60 or args.samples < 1 or any(not 1 <= n <= 2**22 for n in args.sizes):
        parser.error("expected 0 < seconds <= 60, positive samples and sizes in [1, 2^22]")
    if len(set(args.labels)) != 2 or not all(args.labels):
        parser.error("build labels must be distinct and nonempty")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("--output must be a new file in an existing directory")
    operations = OPERATIONS if "all" in args.operations else args.operations
    builds = list(zip(args.labels, (args.baseline.resolve(), args.candidate.resolve())))
    rows = []
    print("kind       size  operation                   baseline ns   candidate ns     change  backends", flush=True)
    cases = itertools.product(dict.fromkeys(args.kinds), dict.fromkeys(args.sizes), dict.fromkeys(operations))
    for index, (kind, size, operation) in enumerate(cases):
        times = {label: [] for label, _ in builds}
        backends = {}
        for repeat in range(args.samples):
            # Alternate which build goes first to spread clock and temperature drift.
            order = builds if (index + repeat) % 2 == 0 else reversed(builds)
            for label, executable in order:
                row = sample(executable, label, kind, operation, size, args.seconds, repeat)
                if backends.setdefault(label, row["backend"]) != row["backend"]:
                    raise ValueError(f"{label}: backend changed during {kind} {size} {operation}")
                rows.append(row)
                times[label].append(float(row["ns_per_call"]))
        baseline, candidate = (median(times[label]) for label in args.labels)
        print(f"{kind:7} {size:7}  {operation:25} {baseline:11.2f} {candidate:14.2f} "
              f"{100 * (candidate / baseline - 1):+9.2f}%  "
              f"{backends[args.labels[0]]} / {backends[args.labels[1]]}", flush=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "xt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
