<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# pffft-neo

pffft-neo is a single-precision FFT library derived from PFFFT and FFTPACK.

## Quality checks

Run `tools/quality.sh` from a source checkout to run the same checks as the
Linux quality job. It checks SPDX headers and retained license texts, formatting,
shell scripts and workflows. It builds and tests the Clang Debug preset, builds
a separate CMake consumer that links `PFFFT::PFFFT` and checks an impulse
transform through the public header, and runs a benchmark smoke test.
It also runs clang-tidy and Cppcheck with the library's compilation database.
The quality job uses Bash 4+, CMake, Ninja, Python 3.8+, C and C++ compilers,
Clang/LLVM (including clang-format and run-clang-tidy), Cppcheck, gersemi,
ShellCheck, shfmt and actionlint. Building the library itself requires only C.

Run `tools/format.sh` to format C, C++, CMake and shell files. Use
`tools/format.sh --check` to check the same files. The generated AVX2 size
table follows its generator's layout. To enable the optional pre-commit hook
in a Git checkout, run `git config --local core.hooksPath .githooks`.
The hook checks staged content, including the staged formatting rules, and
leaves both the index and working files intact.

Run `python3 tools/check_licenses.py` for the SPDX and license-text check alone.
When adding mixed-origin files, record their paths in the checker's `MIXED` set.
Clang-tidy reports integer index widening for review because the FFT kernels
use bounded 32-bit indices. The project's promoted diagnostics fail the quality
check.

For the Linux ASan and UBSan checks, run the following with Clang and Ninja.
The ordinary build owns the instruction-set audit of the compiled library.
The sanitizer preset still runs the audit's assembly fixtures.

```sh
cmake --preset asan-ubsan-debug
cmake --build --preset asan-ubsan-debug --parallel
ctest --preset asan-ubsan-debug --parallel 4
```

Use `tools/cleanup_local.sh --dry-run` to preview local build and analysis output,
then `tools/cleanup_local.sh` to remove it.

## Compare FFT performance

Build the same C++ driver against two library checkouts on the same machine.
Set `PFFFT_SOURCE_DIR` to the baseline source tree and use the current checkout
for the candidate. Use matching CMake options and compilers for both.

```sh
cmake -S benchmarks -B build/bench-base -G Ninja -DCMAKE_BUILD_TYPE=Release -DPFFFT_SOURCE_DIR=/path/to/baseline
cmake -S benchmarks -B build/bench-candidate -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/bench-base
cmake --build build/bench-candidate
python3 tools/bench_compare.py build/bench-base/bench_fft build/bench-candidate/bench_fft --output build/comparison.csv
```

The runner alternates build order, reports median nanoseconds per call and
writes the individual samples, backend and lane count to a new CSV. Positive
percentages mean the candidate took longer. The default covers real and
complex forward transforms and convolution pipelines at sizes 96, 320 and
1024. Use `--sizes`, `--operations all`, `--samples` and `--seconds` to
adjust the sweep. CI validates that the driver runs. Measure performance
on an idle machine.

For dispatch-table research, build the same revision twice, with
`-DPFFFT_ENABLE_WIDE_DISPATCH=OFF` for the baseline and
`-DCMAKE_C_FLAGS=-DPFFFT_WIDE_ALL_SIZES` for the candidate. Run the comparison
with `--labels native4 wide --operations all` and at least three samples per
case. Inspect the CSV with `python3 tools/wide_size_table.py --report build/comparison.csv`.

## Compare FFT accuracy

`benchmarks/` also builds `fft_accuracy`, which measures every transform
against a long-double reference DFT and hashes the results. Configure the two
build trees as for the timing comparison (or reuse them), then run

```sh
cmake --build build/bench-base --target fft_accuracy
cmake --build build/bench-candidate --target fft_accuracy
python3 tools/accuracy_compare.py build/bench-base/fft_accuracy build/bench-candidate/fft_accuracy
```

Both builds get the same inputs: for each real and complex size up to
`--max` (default 65536), 2 seeds (`--seeds`, `--seed-base`) of at least 8192
points each. Each size runs on the native lanes, on the 8-lane backend where
the build and CPU have it (forced, whatever the dispatch tables say), with
the forced depth-first and breadth-first schedules where they apply, and as
`pffft_new_setup()` picks (`--backends`). A case is a backend, kind, size,
direction and ordering. The report lists which cases give the same bits (a
hash of inputs and outputs) and, for the others, the pass factors and the
pooled relative RMS errors of both builds.

The comparison fails on a check within a build (in place, `zreorder`, finite
output, the test suite's error limit), on cases the candidate lost (rows of
a forced schedule, which depend on the plan, only if `--backends` asks for
them), or if the candidate's error over the baseline's exceeds 1.05 at one
case (`--case-tol`) or 1.01 as the geometric mean over the sizes of a
backend, kind and direction, or over its powers of two or its sizes with a
factor 3, 5 or 15 (`--class-tol`). With other seeds a case's ratio moves by
0.4% in the median and at most 3%, a class mean by under 0.3%. `--bits`
requires every case to be bit-identical, for changes that should only move
data; `--hash-only` skips the reference. `--output` writes every case to a
CSV. The default run takes seconds; `--max 1048576`, which reaches the
native depth-first schedule, about 1.5 minutes on 32 threads (`--jobs`).

## License

Copyright (C) 2026 by arancormonk. Starting November 22, 2026, arancormonk's
original contributions to this distribution are licensed under the GNU General
Public License, version 2 or (at your option) any later version
(`GPL-2.0-or-later`). This includes work carried over from the
[PFFFT fork](https://github.com/arancormonk/pffft/tree/2fb266307c0d73bc1efa3b076af6bef8fb517a00).
See [LICENSE](LICENSE) for the full GPLv2 text.

Inherited PFFFT and FFTPACK portions retain their original permissive terms.
Files with both project and upstream contributions carry the SPDX expression
`GPL-2.0-or-later AND LicenseRef-PFFFT`. See the
[upstream notices](third_party/pffft/LicenseRef-PFFFT.txt) for the copyright
notices, redistribution conditions, and disclaimers, including those specific
to the x86 FMA header. The [original PFFFT license](third_party/pffft/LICENSE.txt)
is also included.

When distributing a binary, include the license texts and notices. Standalone
CMake builds collect them in `pffft-neo-licenses/` in the build directory.
One way to meet GPLv2's source requirement is to distribute the complete
matching source tree, including the build scripts, alongside the binary.
