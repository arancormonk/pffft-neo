// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  PFFFT_WIDE: 1 when pffft.c builds the 8-lane AVX2 float backend, and
  PFFFT_PREFETCH (below). Include after simd/pf_float.h, which defines
  PFFFT_SSE_FLOAT_BACKEND.
*/

#ifndef PFFFT_WIDE_CONFIG_H
#define PFFFT_WIDE_CONFIG_H

/*
  On x86-64 the 4-lane SSE implementation can be complemented by the 8-lane
  AVX2 backend of pffft_wide_impl.h, chosen per setup at runtime. GCC and
  Clang compile it with function target pragmas. MSVC, which has no
  per-function targets, compiles it with /arch:AVX2 as its own translation
  unit, pffft_wide_tu.c (PFFFT_WIDE_TU defined for both files), or as part
  of a library built for AVX2 throughout; otherwise it is left out (MSVC
  generates markedly slower AVX2 code without /arch:AVX2). The minimum
  compiler versions on Windows are the oldest ones tested there.
*/
#if defined(PFFFT_WIDE_TU) && defined(_MSC_VER) && !defined(__clang__)
#define PFFFT_WIDE_SEPARATE 1 /* the backend is in pffft_wide_tu.c */
#endif
#if defined(__clang__) && (defined(_WIN32) || defined(__CYGWIN__))
#define PFFFT_WIDE_COMPILER (__clang_major__ >= 19)
#elif defined(__clang__)
#define PFFFT_WIDE_COMPILER (__clang_major__ >= 10)
#elif defined(__GNUC__) && (defined(_WIN32) || defined(__CYGWIN__))
/* Tested with MinGW-w64 GCC 13 and 15. GCC cannot align 32-byte
   stack slots on Windows (GCC bug 54412), which -O0 spills of 256-bit values
   need; GCC 16 stops there with an internal compiler error */
#if defined(__OPTIMIZE__)
#define PFFFT_WIDE_COMPILER (__GNUC__ >= 13)
#else
#define PFFFT_WIDE_COMPILER 0
#endif
#elif defined(__GNUC__)
#define PFFFT_WIDE_COMPILER (__GNUC__ >= 5)
#elif defined(_MSC_VER) && (defined(PFFFT_WIDE_SEPARATE) || defined(__AVX2__))
#define PFFFT_WIDE_COMPILER (_MSC_VER >= 1944) /* Visual Studio 2022 17.14 in CI */
#else
#define PFFFT_WIDE_COMPILER 0
#endif
/* _M_ARM64EC: the MSVC ARM64EC target also defines _M_X64 */
#if !defined(PFFFT_DISABLE_WIDE) && defined(PFFFT_SSE_FLOAT_BACKEND) && PFFFT_WIDE_COMPILER                            \
    && (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_ARM64EC)
#define PFFFT_WIDE 1
#else
#define PFFFT_WIDE 0
#endif

/*
  PFFFT_PREFETCH: 1 when the library has the 4-lane SSE transforms whose
  passes prefetch the lines they write (pffft_passes_impl.h), in a
  translation unit of their own, pffft_prefetch_tu.c, which
  pffft_new_setup() picks on Intel CPUs for transforms out of the level-1
  data cache (pffft_cpu_x86.h). Only in builds with the 8-lane backend,
  whose CPU probe it shares, and that compile pffft_prefetch_tu.c and
  define PFFFT_PREFETCH_TU for it and pffft.c (CMakeLists.txt does both,
  unless PFFFT_DISABLE_PREFETCH); a build of pffft.c alone has none.
*/
#if PFFFT_WIDE && defined(PFFFT_PREFETCH_TU) && !defined(PFFFT_DISABLE_PREFETCH)
#define PFFFT_PREFETCH 1
#else
#define PFFFT_PREFETCH 0
#endif

#endif /* PFFFT_WIDE_CONFIG_H */
