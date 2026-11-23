# SPDX-License-Identifier: GPL-2.0-or-later
cmake_minimum_required(VERSION 3.18)
project(pffft_consumer LANGUAGES C)

add_subdirectory(../.. pffft)
if(TARGET test_pffft_float)
    message(FATAL_ERROR "Embedding PFFFT unexpectedly enabled its own tests")
endif()
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE PFFFT::PFFFT)
enable_testing()
add_test(NAME consumer_impulse COMMAND consumer)
