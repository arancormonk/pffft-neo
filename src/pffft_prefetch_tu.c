// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  The transforms of the native 4-lane SSE implementation whose passes
  prefetch the lines they write (PFFFT_PREFETCH: pffft_wide_config.h,
  pffft_passes_impl.h), which pffft.c's public functions run for a setup
  with PFFFT_SETUP_PF. A translation unit of its own, so that pffft.c
  compiles the code of a build without prefetching. Baseline instruction
  set: compile it with the flags of pffft.c, and define PFFFT_PREFETCH_TU
  for both (CMakeLists.txt does). Without PFFFT_PREFETCH this file has only
  a function that nothing calls.
*/

#define PFFFT_PREFETCH_TU_PART 1
#include "pffft_impl.h"

#if !PFFFT_PREFETCH
/* not empty, as ISO C asks; with instructions, as the instruction audit of
   MSVC objects (tests/isa_audit.cmake) does */
void pffft_prefetch_tu_unused(void);

void
pffft_prefetch_tu_unused(void) {}
#endif
