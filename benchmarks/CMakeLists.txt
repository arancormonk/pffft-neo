# SPDX-License-Identifier: GPL-2.0-or-later
cmake_minimum_required(VERSION 3.18)
project(pffft_benchmarks LANGUAGES C CXX)

# Use the same driver when comparing different library checkouts.
set(PFFFT_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/.."
    CACHE PATH
    "pffft-neo source directory"
)
add_subdirectory("${PFFFT_SOURCE_DIR}" pffft)
add_executable(bench_fft bench_fft.cpp)
target_compile_features(bench_fft PRIVATE cxx_std_11)
target_link_libraries(bench_fft PRIVATE PFFFT::PFFFT)

enable_testing()
add_test(
    NAME benchmark_smoke
    COMMAND bench_fft real pipeline 96 1 0.002
)
