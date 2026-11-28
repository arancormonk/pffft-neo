#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Verdicts of accuracy_compare.py on synthetic fft_accuracy rows."""

import argparse
import contextlib
import io
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accuracy_compare  # noqa: E402

META = {"simd_arch": "SSE1", "simd_size": "4", "w8": "1", "hash_only": "0"}
SIZES = (16, 32, 48, 64, 80, 96)


def rows(backend="native", rms=1e-7):
    """Identical-looking rows of one backend at SIZES."""
    out = {}
    for n, direction, order in itertools.product(SIZES, ("forward", "backward"), ("ordered", "unordered")):
        key = (backend, "complex", n, direction, order)
        out[key] = dict(backend=backend, lanes=4, kind="complex", n=n, direction=direction, order=order,
                        factors="4x4", df="0", inputs=512, rms=rms, max=2 * rms, in_hash=f"in{n}{direction}",
                        hash=f"out{n}{direction}{order}", checks="")
    return out


def verdict(a, b, b_meta=None, **options):
    args = argparse.Namespace(case_tol=0.05, class_tol=0.01, bits=False, verbose=False, backends=None)
    for name, value in options.items():
        setattr(args, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        failed, merged = accuracy_compare.compare(META, a, [], b_meta or META, b, [], args)
    return failed, merged


def change(table, n, direction="forward", **fields):
    for order in ("ordered", "unordered"):
        table[("native", "complex", n, direction, order)].update(fields)


class Verdicts(unittest.TestCase):
    def test_identical_builds_pass(self):
        failed, merged = verdict(rows(), rows(), bits=True)
        self.assertEqual(failed, [])
        self.assertTrue(all(m["identical"] for m in merged))

    def test_small_change_passes_but_is_not_identical(self):
        b = rows()
        change(b, 48, hash="other", rms=1.02e-7)
        failed, merged = verdict(rows(), b)
        self.assertEqual(failed, [])
        self.assertEqual(sum(1 - m["identical"] for m in merged), 2)
        failed, _ = verdict(rows(), b, bits=True)
        self.assertTrue(any("--bits" in f for f in failed))

    def test_case_regression_fails(self):
        b = rows()
        change(b, 48, hash="other", rms=1.06e-7)
        failed, _ = verdict(rows(), b, class_tol=1.0)
        self.assertTrue(any("native complex 48 forward ordered: less accurate" in f for f in failed))

    def test_class_regression_fails(self):
        b = rows()
        for n in SIZES:
            change(b, n, hash="other", rms=1.03e-7)
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("less accurate on average" in f for f in failed))

    def test_family_regression_fails(self):
        b = rows()
        for n in (48, 96):  # the sizes with a factor 3
            change(b, n, hash="other", rms=1.03e-7)
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("3* sizes: less accurate on average" in f for f in failed))

    def test_missing_case_fails(self):
        b = rows()
        del b[("native", "complex", 64, "forward", "ordered")]
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("only in the baseline" in f for f in failed))

    def test_schedule_rows_may_disappear(self):
        a = {**rows(), **rows("native-df")}
        failed, merged = verdict(a, rows())
        self.assertEqual(failed, [])
        self.assertEqual(sum(m["coverage"] == "baseline_only" for m in merged), len(rows("native-df")))
        failed, _ = verdict(a, rows(), backends=["native", "native-df"])
        self.assertTrue(any("only in the baseline" in f for f in failed))

    def test_nothing_compared_fails(self):
        failed, _ = verdict({}, {}, bits=True)
        self.assertTrue(any("no case in both builds" in f for f in failed))

    def test_wide_availability_change_fails(self):
        failed, _ = verdict(rows(), rows(), b_meta={**META, "w8": "0"})
        self.assertTrue(any("8-lane backend" in f for f in failed))

    def test_auto_dispatch_change_keeps_identity(self):
        a, b = rows("auto"), rows("auto")
        for order in ("ordered", "unordered"):
            b[("auto", "complex", 64, "forward", order)]["lanes"] = 8
        failed, merged = verdict(a, b, bits=True)
        self.assertEqual(failed, [])
        self.assertTrue(all(m["identical"] for m in merged))

    def test_requested_backend_without_rows_fails(self):
        failed, _ = verdict(rows(), rows(), backends=["native", "w8"])
        self.assertTrue(any("requested backend w8" in f for f in failed))

    def test_check_failure_fails(self):
        b = rows()
        change(b, 32, checks="inplace")
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("fail a check" in f for f in failed))

    def test_zero_errors(self):
        a, b = rows(), rows()
        change(a, 16, rms=0.0)
        change(b, 16, rms=0.0)
        self.assertEqual(verdict(a, b)[0], [])
        change(b, 16, rms=1e-9, hash="other")
        failed, _ = verdict(a, b)
        self.assertTrue(any("native complex 16 forward ordered: less accurate" in f for f in failed))
        failed, _ = verdict(b, a)
        self.assertEqual(failed, [])

    def test_different_inputs_fail(self):
        b = rows()
        change(b, 80, in_hash="other")
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("different inputs" in f for f in failed))

    def test_lane_change_fails_outside_auto(self):
        b = rows()
        change(b, 64, lanes=8)
        failed, _ = verdict(rows(), b)
        self.assertTrue(any("lanes in the baseline" in f for f in failed))


if __name__ == "__main__":
    unittest.main()
