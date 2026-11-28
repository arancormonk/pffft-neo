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
    with one element per leg (complex factors 2, ..., 4 become ..., 8, or,
    with a 3, 2, ..., 3, ... become ..., 6, a twiddle-free radix-6 pass of
    the 2 and a 3; real factors 2, 4, ..., 3..., 5... become 4..., 3...,
    5..., 8, the 3s and 5s then with an even ido, or, up to
    PFFFT_REAL_RADIX6_MAX_BYTES (..._WIDE with 8 lanes), end with a
    radix-6 pass of the 2 and a 3). One pass fewer over the data; results
    differ in the last bits at the sizes whose plan changes.

  PFFFT_INPLACE_FIRST_PASS
    Complex transforms on the backends with fused layout passes: the first
    pass, which reads and writes the same positions, runs in place when the
    parity of the pass count would otherwise need an extra sweep over the
    data to (de)interleave an in-place transform. Real transforms are not
    changed.

  PFFFT_DEPTH_FIRST
    Complex transforms on the backends with fused layout passes, from
    PFFFT_DEPTH_FIRST_MIN_BYTES (4 lanes; ..._INTEL and ..._AMD on those
    CPUs) or PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE (8 lanes) bytes per buffer
    on, whose plan starts with radix-4 passes:
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

/* PFFFT_REAL_RADIX6_MAX_BYTES, PFFFT_REAL_RADIX6_MAX_BYTES_WIDE: with
   PFFFT_TERMINAL_RADIX8, the largest real transform, in bytes of an array
   (4 N), whose plan merges its lone factor 2 with a 3 into a radix-6 last
   pass instead of with a 4 into a radix-8 one, where it has a 3; the
   first for 4 lanes and the scalar backends, the second for 8 lanes. 0:
   never. Against the radix-8 merge (real forward and backward, every
   such N up to 393216), with 4 lanes the radix-6 one took up to 5% less
   time up to N = 3456 (13.5 KiB) on Haswell (Xeon E5-2697 v3) and, in
   the forward transform, up to 10% more from N = 4320 (17 KiB) on; on
   Zen 3, in cache, 2 to 9% less forward and from 2% more to 10% less
   backward, and out of cache 0 to 3% less. The limit
   is Haswell's crossover. With 8 lanes (Zen 3) it was -2 to +3% forward
   and 0 to 5% less backward in cache: not a clear win, so off. (Both
   lane counts on Zen 3 also ran the backward transform up to 10% faster
   where N/2 bytes is a multiple of 4096: there the 8 output streams of
   radb8_ps, l1 vectors apart, fall into one cache set; radb8_ps now
   writes four of them at a time there, and these measurements predate
   that.) */
#ifndef PFFFT_REAL_RADIX6_MAX_BYTES
#define PFFFT_REAL_RADIX6_MAX_BYTES 16384
#endif
#ifndef PFFFT_REAL_RADIX6_MAX_BYTES_WIDE
#define PFFFT_REAL_RADIX6_MAX_BYTES_WIDE 0
#endif

/* Tuning of PFFFT_DEPTH_FIRST, integer constant expressions too. Measured
   on a Zen 3 core (Ryzen 9 5950X: 32 KiB, 512 KiB and 32 MiB of cache per
   eight cores) and a Haswell one (Xeon E5-2697 v3: 32 KiB, 256 KiB and 35
   MiB), with GCC (on Haswell also Clang), at the powers of two from 2^11
   to 2^22 points and at 3 * 2^k, whose plans end with a radix-6 pass, in
   every complex transform mode: forward and backward, ordered and not, in
   place and not, and a convolution. "Less time" below is the median over
   the modes against the other schedule.

   PFFFT_DEPTH_FIRST_MIN_BYTES, ..._INTEL, ..._AMD: the smallest buffer, 8
   N bytes of a complex transform, for the depth-first schedule with 4
   lanes. pffft_new_setup() takes ..._INTEL on Intel CPUs and ..._AMD on
   AMD ones (CPUID vendor, pffft_cpu_x86.h), PFFFT_DEPTH_FIRST_MIN_BYTES on
   other CPUs, on x86 CPUs whose CPUID stops below leaf 7, and in builds
   without the 8-lane backend, which has the CPU probe. A build that
   defines PFFFT_DEPTH_FIRST_MIN_BYTES sets it for every vendor that it
   does not define a value of its own for: PFFFT_DEPTH_FIRST_MIN_BYTES=0
   makes every complex 4-lane setup depth first where its plan allows. With
   4 lanes a slice is a quarter of the buffer (with 8 a sixteenth), and
   bringing the slices back into natural order costs more than the passes
   gain until the other schedule runs out of a cache level.
   - Haswell: depth first took 4 to 14% less time from 512 KiB to 16 MiB,
     each mode at least 2% less, and a third less at 32 MiB; 3% less at
     256 KiB (one mode 1% more), 2 and 8% more at 64 and 128 KiB, and
     about the same below. The other
     schedule there prefetches its output (PFFFT_PREFETCH,
     pffft_cpu_x86.h); against it without the prefetches, depth first took
     12 to 19% less from 32 KiB on.
   - Zen 3, which does not prefetch: depth first took 2 to 11% more time
     from 128 KiB to 4 MiB, at 8 MiB about the same (backward 6 to 9%
     more), 6% less at 16 MiB and 28% less at 32 MiB; at 3 * 2^k, from
     1.5% less at 48 KiB to 7% more, and the same at 12 MiB.
   - Other CPUs: 8 MiB, the value first measured on Zen 3.

   PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE: the same with 8 lanes, on every CPU
   that runs them (by default only measured AMD ones, pffft_wide_select.h).
   On Zen 3 depth first took 7 to 10% less time from 32 KiB to 2 MiB, every
   mode less, 0.5 to 3% less at 4 and 8 MiB, and 14% more at
   16 KiB; at 3 * 2^k 4 to 7% less from 96 KiB to 1.5 MiB, 1% more at 6
   MiB.

   PFFFT_DEPTH_FIRST_LEAF_BYTES: a leaf is the sub-transform that the last
   passes finish between its part of the buffer and scratch of the same
   size, at least two passes. The depth-first passes go on until the leaf
   is not larger than this, so that leaf and scratch stay in the
   first-level cache. Of leaves from 0.5 to 128 KiB, those of 4 to 16 KiB
   were fastest on Zen 3: smaller ones spend more on calls, larger ones
   leave the cache. On Haswell 16 KiB was fastest too, 8 KiB the same, 4
   KiB 1 to 2% slower, 32 and 64 KiB 5 to 11% slower.

   PFFFT_DEPTH_FIRST_TILE: regions of a slice that are brought into
   natural order together, a power of two. The elements of a region are
   read in sequence and stored 4^deep elements apart (forward; backward
   the reverse), so a tile makes short runs on the one side and bounds the
   streams on the other. Tiles of 4 were fastest; of 16 and 64, up to 20%
   slower with 8 lanes at 2^20 points and several times slower with 4. */
#if defined(PFFFT_DEPTH_FIRST_MIN_BYTES)
/* the build's value, for every vendor that has none of its own */
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#define PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL (PFFFT_DEPTH_FIRST_MIN_BYTES)
#endif
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#define PFFFT_DEPTH_FIRST_MIN_BYTES_AMD (PFFFT_DEPTH_FIRST_MIN_BYTES)
#endif
#else
#define PFFFT_DEPTH_FIRST_MIN_BYTES (8 * 1024 * 1024)
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#define PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL (512 * 1024)
#endif
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#define PFFFT_DEPTH_FIRST_MIN_BYTES_AMD (16 * 1024 * 1024)
#endif
#endif
#ifndef PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE
#define PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE (32 * 1024)
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
#if (PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL) < 0 || (PFFFT_DEPTH_FIRST_MIN_BYTES_AMD) < 0
#error "PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL and ..._MIN_BYTES_AMD must not be negative"
#endif
#if (PFFFT_REAL_RADIX6_MAX_BYTES) < 0 || (PFFFT_REAL_RADIX6_MAX_BYTES_WIDE) < 0
#error "PFFFT_REAL_RADIX6_MAX_BYTES and ..._MAX_BYTES_WIDE must not be negative"
#endif
#if (PFFFT_DEPTH_FIRST_TILE) < 1 || (PFFFT_DEPTH_FIRST_TILE) > 256                                                     \
    || ((PFFFT_DEPTH_FIRST_TILE) & ((PFFFT_DEPTH_FIRST_TILE) - 1)) != 0
#error "PFFFT_DEPTH_FIRST_TILE must be a power of two up to 256"
#endif

#endif /* PFFFT_PASS_CONFIG_H */
