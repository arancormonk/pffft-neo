# SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
add_executable(test_pffft_float test_pffft.c)
add_executable(test_pffft_accuracy_float test_pffft_accuracy.c)
add_executable(test_pffft_wide test_pffft_wide.c)
add_executable(test_fft_factors test_fft_factors.c)
foreach(
    test
    IN
    ITEMS
        test_pffft_float
        test_pffft_accuracy_float
        test_pffft_wide
        test_fft_factors
)
    target_link_libraries(${test} PRIVATE PFFFT)
    target_include_directories(${test} PRIVATE ../src)
    add_test(NAME ${test} COMMAND ${test})
endforeach()
add_test(NAME test_pffft_simd_macros COMMAND test_pffft_float --test-simd)

if(NOT WIN32)
    find_package(Threads)
    if(Threads_FOUND)
        target_compile_definitions(test_pffft_wide PRIVATE PFFFT_TEST_THREADS)
        target_link_libraries(test_pffft_wide PRIVATE Threads::Threads)
    endif()
endif()

# Exercise static pass functions under the same flags as the library.
add_executable(
    test_pffft_passes_float
    test_pffft_passes.c
    ../src/pffft_common.c
)
target_include_directories(
    test_pffft_passes_float
    PRIVATE $<TARGET_PROPERTY:PFFFT,INCLUDE_DIRECTORIES>
)
target_compile_definitions(
    test_pffft_passes_float
    PRIVATE $<TARGET_PROPERTY:PFFFT,COMPILE_DEFINITIONS>
)
target_compile_options(
    test_pffft_passes_float
    PRIVATE $<TARGET_PROPERTY:PFFFT,COMPILE_OPTIONS>
)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_pffft_passes_float PRIVATE m)
endif()
if(TARGET pffft_wide_object)
    target_sources(
        test_pffft_passes_float
        PRIVATE $<TARGET_OBJECTS:pffft_wide_object>
    )
    target_compile_definitions(test_pffft_wide PRIVATE PFFFT_WIDE_TU=1)
endif()
add_test(NAME test_pffft_passes_float COMMAND test_pffft_passes_float)

add_executable(test_pffft_pass_config test_pffft_pass_config.c)
target_include_directories(test_pffft_pass_config PRIVATE ../src)
add_test(NAME test_pffft_pass_config COMMAND test_pffft_pass_config)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_setup_alloc test_setup_alloc.c)
    target_link_libraries(test_setup_alloc PRIVATE PFFFT)
    target_include_directories(test_setup_alloc PRIVATE ../src)
    target_link_options(
        test_setup_alloc
        PRIVATE -Wl,--wrap=malloc -Wl,--wrap=free
    )
    target_compile_options(test_setup_alloc PRIVATE -UNDEBUG -fno-lto)
    add_test(NAME test_setup_alloc COMMAND test_setup_alloc)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
    add_executable(test_simd_unfused test_simd_unfused.c)
    target_include_directories(test_simd_unfused PRIVATE ../src)
    target_compile_definitions(test_simd_unfused PRIVATE PFFFT_DISABLE_FMA=1)
    target_compile_options(test_simd_unfused PRIVATE -ffp-contract=off -UNDEBUG)
    if(PFFFT_USE_SCALAR_VECT)
        target_compile_definitions(
            test_simd_unfused
            PRIVATE PFFFT_SCALVEC_ENABLED=1
        )
    endif()
    if(NOT PFFFT_USE_SIMD)
        target_compile_definitions(
            test_simd_unfused
            PRIVATE PFFFT_SIMD_DISABLE=1
        )
    endif()
    add_test(NAME test_simd_unfused COMMAND test_simd_unfused)
    set_tests_properties(test_simd_unfused PROPERTIES SKIP_RETURN_CODE 77)
endif()

find_package(Python3 3.8 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME wide_size_table_selftest
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/wide_size_table.py --self-test
    )
    include(isa_audit.cmake)
endif()
