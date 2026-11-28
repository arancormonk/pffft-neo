# SPDX-License-Identifier: GPL-2.0-or-later
# The instruction audit understands ELF and PE/COFF for baseline x86-64.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32))
    return()
endif()
include(CheckCSourceCompiles)
unset(PFFFT_BASELINE_X64 CACHE)
check_c_source_compiles(
    "
  #if !(defined(__x86_64__) || defined(_M_X64)) || defined(_M_ARM64EC) || defined(__SSE3__) || defined(__AVX__)
  #error requires a baseline x86-64 compiler target
  #endif
  int main(void) { return 0; }
"
    PFFFT_BASELINE_X64
)
if(NOT PFFFT_BASELINE_X64)
    return()
endif()

# Prefer the toolchain's disassembler when it is GNU objdump. LLVM objdump
# uses different output conventions, including in Visual Studio toolchains.
if(WIN32)
    find_program(PFFFT_GNU_OBJDUMP NAMES x86_64-w64-mingw32-objdump objdump)
else()
    find_program(PFFFT_GNU_OBJDUMP NAMES objdump)
endif()
foreach(tool IN ITEMS "${CMAKE_OBJDUMP}" "${PFFFT_GNU_OBJDUMP}")
    if(tool)
        execute_process(
            COMMAND "${tool}" --version
            OUTPUT_VARIABLE version
            ERROR_QUIET
        )
        if(version MATCHES "^GNU objdump")
            set(objdump "${tool}")
            break()
        endif()
    endif()
endforeach()
if(NOT objdump)
    message(
        STATUS
        "Skipping x86-64 instruction audit (GNU objdump unavailable)"
    )
    return()
endif()

set(audit
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/check_isa_baseline.py"
)
set(args --objdump "${objdump}" $<$<CONFIG:Debug>:--no-transitions>)
if(PFFFT_DISABLE_FMA OR CMAKE_C_FLAGS MATCHES "PFFFT_DISABLE_FMA")
    list(APPEND args --no-fma)
endif()
if(MSVC)
    target_link_options(test_pffft_wide PRIVATE /MAP)
    set(objects $<TARGET_OBJECTS:PFFFT>)
    if(TARGET pffft_prefetch_object)
        list(APPEND objects $<TARGET_OBJECTS:pffft_prefetch_object>)
    endif()
    set(required "")
    if(TARGET pffft_wide_object)
        list(APPEND objects $<TARGET_OBJECTS:pffft_wide_object>)
        list(APPEND args --wide-tu)
        set(required pffft_w8_new_setup,pffft_w8_transform)
    endif()
    add_test(
        NAME check_isa_baseline
        COMMAND ${audit} ${args} ${objects} --skip-empty --all-il-ok
    )
    add_test(
        NAME check_isa_baseline_image
        COMMAND
            ${audit} ${args} --map-objects "^(pffft:)?[^:<]+$" --require
            "${required}" --require-unless-ltcg pffft_new_setup,pffft_transform
            --ltcg-objects "${objects}" $<TARGET_FILE:test_pffft_wide> --map
            $<TARGET_FILE_DIR:test_pffft_wide>/test_pffft_wide.map
    )
    find_program(PFFFT_AUDIT_CLANG NAMES clang)
    if(PFFFT_AUDIT_CLANG)
        set(assembler
            "${PFFFT_AUDIT_CLANG}"
            --cc-arg
            --target=x86_64-pc-windows-msvc
            --linker
            "${CMAKE_LINKER}"
        )
    endif()
else()
    # An IPO archive contains bitcode, so is one built with -flto in the C
    # flags. Audit the linked image in that case.
    get_target_property(ipo PFFFT INTERPROCEDURAL_OPTIMIZATION)
    set(lto_flag "(^| )-flto")
    set(image "$<TARGET_FILE:PFFFT>")
    # an archive must have pffft.c's code, so that pffft_prefetch_tu.c's code
    # alone cannot pass (in a linked image these may be inlined)
    set(require "--require;pffft_new_setup$<COMMA>pffft_transform")
    if(ipo OR CMAKE_C_FLAGS MATCHES "${lto_flag}")
        set(image "$<TARGET_FILE:test_pffft_wide>")
        set(require "")
    endif()
    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
        string(TOUPPER "${config}" upper)
        get_property(
            config_ipo
            TARGET PFFFT
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_${upper}
        )
        if(config_ipo OR CMAKE_C_FLAGS_${upper} MATCHES "${lto_flag}")
            set(image
                "$<IF:$<CONFIG:${config}>,$<TARGET_FILE:test_pffft_wide>,${image}>"
            )
            set(require "$<IF:$<CONFIG:${config}>,,${require}>")
        endif()
    endforeach()
    add_test(
        NAME check_isa_baseline
        COMMAND ${audit} ${args} "${require}" ${image}
        COMMAND_EXPAND_LISTS
    )
    set(assembler "${CMAKE_C_COMPILER}")
endif()
if(assembler)
    add_test(
        NAME check_isa_baseline_selftest
        COMMAND ${audit} --self-test --objdump "${objdump}" --cc ${assembler}
    )
endif()
