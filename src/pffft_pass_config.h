// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Compile-time switches of the pass schedule (pffft_passes_impl.h), each 0
  or 1. They apply to every backend built in a translation unit, including
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

  PFFFT_DEPTH_FIRST
    Complex transforms on the backends with fused layout passes, from
    PFFFT_DEPTH_FIRST_MIN_BYTES (4 lanes) or PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE
    (8 lanes) bytes per buffer on, whose plan starts with radix-4 passes:
    those passes run in place and depth first, so that a sub-transform is
    finished while it is in the cache, where the other schedule sweeps the
    whole buffer once per pass. The last passes, on sub-transforms of at
    most PFFFT_DEPTH_FIRST_LEAF_BYTES bytes, still alternate between two
    buffers (pffft_passes_impl.h has the details). Spectra keep their
    layout, real transforms are not changed. Forward results keep their
    bits, backward ones differ in the last bits.

  Defaults: all 1 on every target and backend, except the terminal
  radix-8 pass on the single-lane scalar build (SIMD_SZ == 1), where it is
  0: complex transforms measured slower with it there, while real
  transforms gained. Define PFFFT_TERMINAL_RADIX8=1 to enable it there.
  The in-place first pass and the depth-first passes apply to the
  four-lane and eight-lane backends.

  The compiler may define any switch to an integer constant expression
  evaluating to 0 or 1. This header normalizes it to that literal value.

  Included by pffft_passes_impl.h, after the simd/pf_*.h backend selection,
  which defines SIMD_SZ, and read once per translation unit: the 8-lane
  backend, built only with the 4-lane SSE float one, gets the same values.
*/

#ifndef PFFFT_PASS_CONFIG_H
#define PFFFT_PASS_CONFIG_H

/* The defaults: all 1, except the terminal radix-8 pass on the
   single-lane scalar build (complex transforms measured slower with it
   there; see above); the build may define any of them beforehand. */
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
#ifndef PFFFT_DEPTH_FIRST
#define PFFFT_DEPTH_FIRST 1
#endif

/* Tuning of PFFFT_DEPTH_FIRST, integer constant expressions too. The
   defaults are the values measured fastest on a Zen 3 core (32 KiB, 512
   KiB and 16 MiB of cache) with GCC, over the powers of two from 2^12 to
   2^22 points and every transform mode: forward and backward, ordered and
   not, in place and not.

   PFFFT_DEPTH_FIRST_MIN_BYTES, PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE: the
   smallest buffer, 8 N bytes of a complex transform, for the depth-first
   schedule with 4 and with 8 lanes. With 8 lanes it took 4 to 15% less
   time than the other schedule for buffers of 32 KiB to 2 MiB, between 2%
   less and 7% more at 4 MiB, where the buffers and tables of both
   schedules just fill the last cache level, a quarter to a third less at
   8 MiB and half from 16 MiB on; the default leaves a margin for other
   caches. With 4 lanes the slices are four times as large (a quarter of
   the buffer, not a sixteenth), and bringing them back into natural order
   costs more than the passes gain, 2 to 10% in all, until the buffers
   leave the last cache level: 7 to 17% less time at 8 MiB, 40% less from
   16 MiB on.

   PFFFT_DEPTH_FIRST_LEAF_BYTES: a leaf is the sub-transform that the last
   passes finish between its part of the buffer and scratch of the same
   size, at least two passes. The depth-first passes go on until the leaf
   is not larger than this, so that leaf and scratch stay in the
   first-level cache. Of leaves from 0.5 to 128 KiB, those of 4 to 16 KiB
   were fastest: smaller ones spend more on calls, larger ones leave the
   cache.

   PFFFT_DEPTH_FIRST_TILE: regions of a slice that are brought into
   natural order together, a power of two. The elements of a region are
   read in sequence and stored 4^deep elements apart (forward; backward
   the reverse), so a tile makes short runs on the one side and bounds the
   streams on the other. Tiles of 4 were fastest; of 16 and 64, up to 20%
   slower with 8 lanes at 2^20 points and several times slower with 4. */
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES
#define PFFFT_DEPTH_FIRST_MIN_BYTES (8 * 1024 * 1024)
#endif
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE
#define PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE (128 * 1024)
#endif
#ifndef PFFFT_DEPTH_FIRST_LEAF_BYTES
#define PFFFT_DEPTH_FIRST_LEAF_BYTES 16384
#endif
#ifndef PFFFT_DEPTH_FIRST_TILE
#define PFFFT_DEPTH_FIRST_TILE 4
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
#if (PFFFT_DEPTH_FIRST) != 0 && (PFFFT_DEPTH_FIRST) != 1
#error "PFFFT_DEPTH_FIRST must be 0 or 1"
#endif
#if (PFFFT_DEPTH_FIRST)
#undef PFFFT_DEPTH_FIRST
#define PFFFT_DEPTH_FIRST 1
#else
#undef PFFFT_DEPTH_FIRST
#define PFFFT_DEPTH_FIRST 0
#endif
#if (PFFFT_DEPTH_FIRST_MIN_BYTES) < 0 || (PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE) < 0 || (PFFFT_DEPTH_FIRST_LEAF_BYTES) < 0
#error "PFFFT_DEPTH_FIRST_MIN_BYTES, ..._MIN_BYTES_WIDE and ..._LEAF_BYTES must not be negative"
#endif
#if (PFFFT_DEPTH_FIRST_TILE) < 1 || (PFFFT_DEPTH_FIRST_TILE) > 256                                                     \
    || ((PFFFT_DEPTH_FIRST_TILE) & ((PFFFT_DEPTH_FIRST_TILE) - 1)) != 0
#error "PFFFT_DEPTH_FIRST_TILE must be a power of two up to 256"
#endif

#endif /* PFFFT_PASS_CONFIG_H */
