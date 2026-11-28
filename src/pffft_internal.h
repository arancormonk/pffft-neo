// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Library-internal entry points of pffft.c, for the tests. Not installed and
  not part of the stable API.
*/

#ifndef PFFFT_INTERNAL_H
#define PFFFT_INTERNAL_H

#include <stdio.h>
#include "pffft.h"

#ifdef __cplusplus
extern "C" {
#endif

PFFFT_EXPORT void validate_pffft_simd(void);
PFFFT_EXPORT int validate_pffft_simd_ex(FILE* out);

#define PFFFT_BACKEND_AUTO          0 /* what pffft_new_setup() picks */
#define PFFFT_BACKEND_NATIVE        1 /* the pffft_simd_size()-lane implementation */
#define PFFFT_BACKEND_W8            2 /* the 8-lane AVX2 backend; NULL when unavailable */

/* added to a backend: the pass schedule of a complex setup, which
   pffft_new_setup() picks from the transform size (PFFFT_DEPTH_FIRST in
   pffft_pass_config.h). PFFFT_BACKEND_DEPTH_FIRST applies it wherever the
   plan allows it, at any size and with the smallest leaves, so that small
   transforms run every step of it; PFFFT_BACKEND_BREADTH_FIRST never does.
   The two schedules give the same forward results, bit for bit. */
#define PFFFT_BACKEND_DEPTH_FIRST   0x100
#define PFFFT_BACKEND_BREADTH_FIRST 0x200

/* added to a backend: whether the passes of a native setup prefetch the
   lines they write, which pffft_new_setup() picks from the CPU and the
   transform size where the build has the prefetching passes (x86-64, SSE).
   PFFFT_BACKEND_PREFETCH makes the passes prefetch, at any size and on any
   CPU; PFFFT_BACKEND_NO_PREFETCH never. No effect on the results, nor on
   8-lane setups, depth-first ones, nor in builds without them. */
#define PFFFT_BACKEND_PREFETCH      0x400
#define PFFFT_BACKEND_NO_PREFETCH   0x800

/*
  pffft_new_setup() with a forced backend. PFFFT_BACKEND_W8 ignores the
  CPU-model and size tables (but not the instruction-set checks) so tests
  can run the 8-lane backend on any CPU that supports it. Returns NULL
  where pffft_new_setup() would, or when the requested backend is not
  built in, not supported by the CPU, or does not support N.
*/
PFFFT_EXPORT PFFFT_Setup* pffft_internal_new_setup(int N, pffft_transform_t transform, int backend);

/* 0 for a setup whose passes all sweep the buffer (real setups, small
   ones, plans without leading radix-4 passes, PFFFT_DEPTH_FIRST off), else
   1 + the number of depth-first passes within a slice */
PFFFT_EXPORT int pffft_internal_depth_first_levels(const PFFFT_Setup* setup);

/* 1 if the transforms of the setup run the passes that prefetch the lines
   they write, else 0 (also for 8-lane setups and in builds without them) */
PFFFT_EXPORT int pffft_internal_prefetch_passes(const PFFFT_Setup* setup);

/* the radices of the setup's passes in the order the plan stores them
   (ifac), which is not always the order they run in (real forward
   transforms run them in reverse, depth-first ones by slice). *count gets
   their number. PFFFT_INTERNAL_HAS_PLAN tells a tool built against several
   revisions of the library that it is there. */
#define PFFFT_INTERNAL_HAS_PLAN 1
PFFFT_EXPORT const int* pffft_internal_plan(const PFFFT_Setup* setup, int* count);

#ifdef __cplusplus
}
#endif

#endif /* PFFFT_INTERNAL_H */
