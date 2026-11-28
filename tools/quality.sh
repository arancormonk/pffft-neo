#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
set -euo pipefail
cd "$(dirname "$0")/.."
shopt -s globstar nullglob

python3 tools/check_licenses.py
tools/format.sh --check
shellcheck tools/*.sh tests/*.sh .githooks/pre-commit
actionlint .github/workflows/*.yml

cmake --preset quality
cmake --build build/quality --parallel 4
ctest --parallel 4 --test-dir build/quality --output-on-failure --no-tests=error
cmake -S tests/consumer -B build/consumer -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/consumer --parallel 4
ctest --test-dir build/consumer --output-on-failure --no-tests=error
cmake -S benchmarks -B build/bench-smoke -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/bench-smoke --parallel 4
ctest --test-dir build/bench-smoke --output-on-failure --no-tests=error
clang-tidy --quiet -p build/consumer tests/consumer/main.c
clang-tidy --quiet -p build/bench-smoke benchmarks/bench_fft.cpp
run-clang-tidy -quiet -j 4 -p build/quality
# Cppcheck selects the portable GNU C implementation of the shared code.
cppcheck --project=build/quality/compile_commands.json -D__GNUC__=4 --std=c99 \
  --enable=warning,performance,portability --error-exitcode=1 -j4
# pffft_prefetch_tu.c has code only with the 8-lane backend (x86-64, GCC 5+)
cppcheck --project=build/quality/compile_commands.json --file-filter=src/pffft_prefetch_tu.c -D__GNUC__=5 \
  -D__x86_64__=1 --std=c99 --enable=warning,performance,portability --error-exitcode=1 -j4
