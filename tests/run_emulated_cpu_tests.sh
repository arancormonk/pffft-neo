#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#
# Usage: tests/run_emulated_cpu_tests.sh BUILD_DIR
# Use an x86-64 baseline userspace (including its static libc), GNU objdump,
# Python 3 and qemu-x86_64. Newer distro libraries can fault before main.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$1"
top=$(cd "$1" && pwd)
qemu=${QEMU:-qemu-x86_64}

for config in plain zero-regs; do
  flags="-march=x86-64"
  if [[ $config == zero-regs ]]; then
    # Compiler-generated register clearing must also obey the CPU check.
    flags+=" -fzero-call-used-regs=all"
  fi
  build="$top/$config"
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$flags" -DCMAKE_EXE_LINKER_FLAGS=-static
  cmake --build "$build" --parallel --target test_pffft_float test_pffft_accuracy_float
  ctest --test-dir "$build" -R '^check_isa_baseline$' --output-on-failure --no-tests=error

  cd "$build/tests"
  for cpu in Nehalem SandyBridge Haswell; do
    # Exercise default dispatch, then compare native and available wide
    # transforms against the independent DFT. CTest covers the full size table.
    log="$build/$cpu.log"
    if ! "$qemu" -cpu "$cpu" ./test_pffft_float > "$log" 2>&1 ||
      ! "$qemu" -cpu "$cpu" ./test_pffft_accuracy_float 1024 >> "$log" 2>&1; then
      tail -20 "$log"
      exit 1
    fi
    expected="8-lane backend: not available"
    if [[ $cpu == Haswell ]]; then expected="8-lane backend: tested"; fi
    if ! grep -F "$expected" "$log"; then
      echo "Unexpected backend on $cpu ($config)"
      tail -10 "$log"
      exit 1
    fi
    echo "Passed $cpu ($config)"
  done
done
