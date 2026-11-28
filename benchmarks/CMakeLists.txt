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

# The accuracy A/B driver; tools/accuracy_compare.py runs two builds of it.
# Its reference and helpers come from this tree's tests, the internal header
# from the library it is built against.
add_executable(fft_accuracy fft_accuracy.c)
target_compile_features(fft_accuracy PRIVATE c_std_99)
target_include_directories(
    fft_accuracy
    PRIVATE "${PFFFT_SOURCE_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}/../tests"
)
target_link_libraries(fft_accuracy PRIVATE PFFFT::PFFFT)
add_test(
    NAME accuracy_smoke
    COMMAND fft_accuracy --max 512 --seeds 1
)
find_package(Python3 3.8 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(tools "${CMAKE_CURRENT_LIST_DIR}/../tools")
    add_test(
        NAME accuracy_compare_smoke
        COMMAND
            Python3::Interpreter "${tools}/accuracy_compare.py"
            $<TARGET_FILE:fft_accuracy> $<TARGET_FILE:fft_accuracy> --bits --max
            1024 --jobs 2
    )
    add_test(
        NAME accuracy_compare_selftest
        COMMAND Python3::Interpreter "${tools}/test_accuracy_compare.py"
    )
endif()
