# SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
cmake_minimum_required(VERSION 3.18)
project(pffft-neo LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(pffft_standalone OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(pffft_standalone ON)
    foreach(
        notice
        IN
        ITEMS
            README.md
            LICENSE
            third_party/pffft/LICENSE.txt
            third_party/pffft/LicenseRef-PFFFT.txt
    )
        configure_file("${notice}" "pffft-neo-licenses/${notice}" COPYONLY)
    endforeach()
endif()
option(PFFFT_BUILD_TESTS "Build the float FFT tests" ${pffft_standalone})
option(
    PFFFT_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    ${pffft_standalone}
)
option(PFFFT_USE_SIMD "Use SIMD when supported by the target" ON)
option(
    PFFFT_USE_SCALAR_VECT
    "Use four scalar lanes when hardware SIMD is unavailable"
    ON
)
option(
    PFFFT_ENABLE_WIDE_DISPATCH
    "Build the runtime-selected x86-64 AVX2 backend"
    ON
)
option(PFFFT_DISABLE_FMA "Disable explicit FMA and compiler contraction" OFF)

if(MSVC)
    add_compile_options(/W4 /wd4244 /wd4305)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
    if(PFFFT_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wformat=2
        -Wstrict-prototypes
        -Wmissing-prototypes
        -Wold-style-definition
        -Wvla
    )
    if(PFFFT_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

add_library(PFFFT STATIC src/pffft.c src/pffft_common.c)
add_library(PFFFT::PFFFT ALIAS PFFFT)
set_target_properties(
    PFFFT
    PROPERTIES
        OUTPUT_NAME pffft
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
)
target_include_directories(PFFFT PUBLIC include include/pffft PRIVATE src)
target_compile_definitions(PFFFT PUBLIC PFFFT_STATIC_DEFINE)
if(UNIX AND NOT APPLE)
    target_link_libraries(PFFFT PUBLIC m)
endif()

if(NOT PFFFT_USE_SIMD)
    target_compile_definitions(PFFFT PRIVATE PFFFT_SIMD_DISABLE=1)
endif()
if(PFFFT_USE_SCALAR_VECT)
    target_compile_definitions(PFFFT PRIVATE PFFFT_SCALVEC_ENABLED=1)
endif()
if(NOT PFFFT_ENABLE_WIDE_DISPATCH)
    target_compile_definitions(PFFFT PRIVATE PFFFT_DISABLE_WIDE=1)
endif()
if(PFFFT_DISABLE_FMA)
    target_compile_definitions(PFFFT PRIVATE PFFFT_DISABLE_FMA=1)
    if(MSVC)
        target_compile_options(PFFFT PRIVATE /fp:strict)
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(PFFFT PRIVATE -ffp-contract=off)
    endif()
endif()

# MSVC needs a separate AVX2 object. Keep /GL off for that object so IPO
# cannot move instructions from it into the baseline CPU detection code.
if(
    CMAKE_C_COMPILER_ID STREQUAL "MSVC"
    AND PFFFT_USE_SIMD
    AND PFFFT_ENABLE_WIDE_DISPATCH
)
    include(CheckCSourceCompiles)
    check_c_source_compiles(
        "
    #if !defined(_M_X64) || defined(_M_ARM64EC)
    #error requires an x64 target
    #endif
    int main(void) { return 0; }
  "
        PFFFT_MSVC_X64
    )
    if(PFFFT_MSVC_X64)
        add_library(pffft_wide_object OBJECT src/pffft_wide_tu.c)
        set_target_properties(
            pffft_wide_object
            PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF
        )
        foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
            string(TOUPPER "${config}" upper)
            set_property(
                TARGET pffft_wide_object
                PROPERTY INTERPROCEDURAL_OPTIMIZATION_${upper} OFF
            )
        endforeach()
        target_compile_definitions(PFFFT PRIVATE PFFFT_WIDE_TU=1)
        target_compile_definitions(
            pffft_wide_object
            PRIVATE $<TARGET_PROPERTY:PFFFT,COMPILE_DEFINITIONS>
        )
        target_include_directories(
            pffft_wide_object
            PRIVATE $<TARGET_PROPERTY:PFFFT,INCLUDE_DIRECTORIES>
        )
        target_compile_options(
            pffft_wide_object
            PRIVATE $<TARGET_PROPERTY:PFFFT,COMPILE_OPTIONS>
        )
        set_source_files_properties(
            src/pffft_wide_tu.c
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/GL-"
        )
        target_sources(PFFFT PRIVATE $<TARGET_OBJECTS:pffft_wide_object>)
    endif()
endif()

if(PFFFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
