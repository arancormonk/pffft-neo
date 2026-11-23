#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
set -euo pipefail

usage() {
  cat << 'USAGE'
Usage: tools/cleanup_local.sh [--dry-run]

Remove build/, root build_* CMake trees, compile_commands.json and Python
caches under tools/ and tests/. Paths containing tracked files are preserved.
Use --dry-run to preview the paths, or --help to show this text.
USAGE
}

dry_run=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."
in_worktree=0
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  in_worktree=1
elif [[ -e .git || -L .git ]]; then
  echo "Cannot check tracked paths in this worktree." >&2
  exit 1
fi

shopt -s nullglob
paths=()
if [[ -d build ]]; then paths+=(build); fi
for path in build_*; do
  if [[ -d "$path/CMakeFiles" || -f "$path/CMakeCache.txt" ]]; then paths+=("$path"); fi
done
if [[ -f compile_commands.json || -L compile_commands.json ]]; then paths+=(compile_commands.json); fi
while IFS= read -r -d '' path; do
  paths+=("$path")
done < <(find tools tests -type d -name __pycache__ -prune -print0)

if ((${#paths[@]} == 0)); then echo "Nothing to clean."; fi
for path in "${paths[@]}"; do
  if ((in_worktree)); then
    tracked=$(git --literal-pathspecs ls-files -- "$path")
    if [[ -n $tracked ]]; then
      echo "Skipping tracked path: $path"
      continue
    fi
  fi
  if ((dry_run)); then
    echo "Would remove: $path"
  else
    echo "Removing: $path"
    rm -rf -- "$path"
  fi
done
