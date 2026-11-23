// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Compile-time switches of the pass schedule (pffft_passes_impl.h), each 0
  or 1. Both apply to every backend built in a translation unit, including
  the 8-lane AVX2 one.

  PFFFT_TERMINAL_RADIX8
    No stand-alone radix-2 pass: where a plan has a radix-2 and a radix-4
    pass, the pair is replaced by one twiddle-free radix-8 pass, the one
    with one element per leg (complex factors 2, ..., 4 become ..., 8; real
    power-of-two factors 2, 4, ..., 4 become 4, ..., 8). One pass fewer
    over the data; results differ in the last bits at the sizes whose plan
    changes.

  PFFFT_INPLACE_FIRST_PASS
    Complex transforms on the backends with fused layout passes: the first
    pass, which reads and writes the same positions, runs in place when the
    parity of the pass count would otherwise need an extra sweep over the
    data to (de)interleave an in-place transform. Real transforms are not
    changed.

  Defaults: both 1 on every target and backend, except the terminal
  radix-8 pass on the single-lane scalar build (SIMD_SZ == 1), where it is
  0: complex transforms measured slower with it there, while real
  transforms gained. Define PFFFT_TERMINAL_RADIX8=1 to enable it there.
  The in-place first pass applies to the four-lane and eight-lane backends.

  The compiler may define either switch to an integer constant expression
  evaluating to 0 or 1. This header normalizes it to that literal value.

  Included by pffft_passes_impl.h, after the simd/pf_*.h backend selection,
  which defines SIMD_SZ, and read once per translation unit: the 8-lane
  backend, built only with the 4-lane SSE float one, gets the same values.
*/

#ifndef PFFFT_PASS_CONFIG_H
#define PFFFT_PASS_CONFIG_H

/* The defaults: both 1, except the terminal radix-8 pass on the
   single-lane scalar build (complex transforms measured slower with it
   there; see above); the build may define either beforehand. */
#ifndef SIMD_SZ
#error "pffft_pass_config.h needs SIMD_SZ: include the simd/pf_*.h backend first"
#endif
#ifndef PFFFT_TERMINAL_RADIX8
#if SIMD_SZ == 1
#define PFFFT_TERMINAL_RADIX8 0
#else
#define PFFFT_TERMINAL_RADIX8 1
#endif
#endif
#ifndef PFFFT_INPLACE_FIRST_PASS
#define PFFFT_INPLACE_FIRST_PASS 1
#endif

/* A definition of the build may be any expression, without parentheses
   (-DPFFFT_TERMINAL_RADIX8=0?0:1): checked in parentheses, then redefined
   as the literal 0 or 1, so that every later use (#if X && ..., and the
   tests) sees one token and needs no parentheses of its own. */
#if (PFFFT_TERMINAL_RADIX8) != 0 && (PFFFT_TERMINAL_RADIX8) != 1
#error "PFFFT_TERMINAL_RADIX8 must be 0 or 1"
#endif
#if (PFFFT_TERMINAL_RADIX8)
#undef PFFFT_TERMINAL_RADIX8
#define PFFFT_TERMINAL_RADIX8 1
#else
#undef PFFFT_TERMINAL_RADIX8
#define PFFFT_TERMINAL_RADIX8 0
#endif
#if (PFFFT_INPLACE_FIRST_PASS) != 0 && (PFFFT_INPLACE_FIRST_PASS) != 1
#error "PFFFT_INPLACE_FIRST_PASS must be 0 or 1"
#endif
#if (PFFFT_INPLACE_FIRST_PASS)
#undef PFFFT_INPLACE_FIRST_PASS
#define PFFFT_INPLACE_FIRST_PASS 1
#else
#undef PFFFT_INPLACE_FIRST_PASS
#define PFFFT_INPLACE_FIRST_PASS 0
#endif

#endif /* PFFFT_PASS_CONFIG_H */
