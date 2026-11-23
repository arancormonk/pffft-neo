#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Check project SPDX headers, sidecars and retained notices in a source checkout."""

import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "GPL-2.0-or-later"
MIXED = {
    ".github/workflows/c-cpp.yml",
    "CMakeLists.txt",
    "include/pffft/pffft.h",
    "src/pffft_common.c",
    "src/pffft_cplx_impl.h",
    "src/pffft_impl.h",
    "src/pffft_passes_impl.h",
    "src/pffft_priv_impl.h",
    "src/pffft_real_impl.h",
    "src/pffft_setup.h",
    "src/pffft_wide_impl.h",
    "src/simd/pf_float.h",
    "src/simd/pf_fma_x86.h",
    "src/simd/pf_neon_float.h",
    "src/simd/pf_scalar_float.h",
    "src/simd/pf_sse1_float.h",
    "tests/CMakeLists.txt",
    "tests/test_fft_factors.c",
    "tests/test_pffft.c",
}
NOTICES = {
    "LICENSE": "640d90f6ee401241ff62834d0d1b33f83049e99b8bfdfa04ed1b0a1635dde3e7",
    "third_party/pffft/LICENSE.txt": "6c99fce38c991ff104307ecebe20ff014b5e1f5d9e3426da292da8417c7a78cd",
    "third_party/pffft/LicenseRef-PFFFT.txt": "c2bc1b84e8eb267b46fc25cc62bab7db5ba6b0ad538ed50e802cfd6b139fb2b0",
}
HEADER = re.compile(r"(?:(?://|#|/\*|<!--)\s*)?SPDX-License-Identifier:\s*(.*?)\s*(?:\*/|-->)?")


def project_files():
    paths = {ROOT / name for name in (
        "CMakeLists.txt", "CMakePresets.json", "README.md", ".clang-format",
        ".clang-format-ignore", ".clang-tidy", ".gitignore",
    )}
    for directory in ("include", "src", "tests", "tools", "benchmarks", ".github", ".githooks"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".license":
                paths.add(path)
    return sorted(paths)


def main():
    errors = []
    files = project_files()
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        expected = PROJECT + " AND LicenseRef-PFFFT" if name in MIXED else PROJECT
        header = path.with_suffix(".json.license") if path.suffix == ".json" else path
        if not path.is_file() or not header.is_file():
            errors.append(f"{name}: missing file or SPDX sidecar")
            continue
        with header.open(encoding="utf-8") as stream:
            first, second = stream.readline().strip(), stream.readline().strip()
        line = second if first.startswith(("#!", "SPDX-FileCopyrightText:")) else first
        match = HEADER.fullmatch(line)
        if not match or match[1] != expected:
            errors.append(f"{name}: expected SPDX-License-Identifier: {expected}")
    for name, expected in NOTICES.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            errors.append(f"{name}: missing or changed license text")
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"Verified SPDX headers for {len(files)} files and {len(NOTICES)} license texts.")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
