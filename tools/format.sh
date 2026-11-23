#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
set -euo pipefail
cd "$(dirname "$0")/.."
shopt -s globstar nullglob

mode=${1:---write}
if (($# > 1)); then mode=invalid; fi
case "$mode" in
  --write | --check) ;;
  --staged)
    # Check the index's files and formatting rules, even for a partial commit.
    snapshot=$(mktemp -d)
    trap 'rm -rf -- "$snapshot"' EXIT
    git diff --cached --name-only -z --diff-filter=ACMR > "$snapshot/paths"
    [[ -s "$snapshot/paths" ]] || exit 0
    declare -A staged
    while IFS= read -r -d '' path; do staged["$path"]=1; done < "$snapshot/paths"
    mkdir "$snapshot/tree"
    git checkout-index --all --prefix="$snapshot/tree/"
    cd "$snapshot/tree"
    ;;
  *)
    echo "Usage: tools/format.sh [--write|--check|--staged]" >&2
    exit 2
    ;;
esac

format() {
  local tool=$1 path
  shift
  local files=()
  for path in "$@"; do
    if [[ $mode == --staged && ! ${staged["$path"]+changed} ]]; then continue; fi
    [[ -f "$path" && ! -L "$path" ]] && files+=("$path")
  done
  if ((${#files[@]} == 0)); then return 0; fi
  local args=()
  case "$tool/$mode" in
    clang-format/--write) args=(-i) ;;
    clang-format/*) args=(--dry-run --Werror) ;;
    gersemi/--write) args=(-i) ;;
    gersemi/*) args=(--check --diff) ;;
    shfmt/--write) args=(-w -i 2 -ci -sr) ;;
    shfmt/*) args=(-d -i 2 -ci -sr) ;;
  esac
  "$tool" "${args[@]}" "${files[@]}"
}

format clang-format {include,src,tests,benchmarks}/**/*.{c,h,cpp}
format gersemi CMakeLists.txt {tests,benchmarks}/**/CMakeLists.txt tests/**/*.cmake
format shfmt tools/**/*.sh tests/**/*.sh .githooks/pre-commit
