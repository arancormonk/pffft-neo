// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Layout of PFFFT_Setup, shared by pffft_priv_impl.h and the 8-lane
  backend when MSVC compiles that as its own
  translation unit (pffft_wide_tu.c), so that both sides agree on it.
*/

#ifndef PFFFT_SETUP_H
#define PFFFT_SETUP_H

/* largest accepted N: larger values overflow int in the setup and passes */
#define PFFFT_MAX_N (1 << 26)

#ifndef IFAC_MAX_SIZE
#define IFAC_MAX_SIZE 25 /* max number of integer factors for the decomposition, +2 */
#endif

#define PFFFT_SETUP_KIND_NATIVE 0 /* the SIMD_SZ-lane implementation of pffft_priv_impl.h */
#define PFFFT_SETUP_KIND_W8     1 /* 8-lane AVX2 float backend, pffft_wide_impl.h */
/* added to PFFFT_SETUP_KIND_NATIVE in 'kind' (PFFFT_PREFETCH): the
   transforms run the passes that prefetch the lines they write
   (pffft_prefetch_tu.c) */
#define PFFFT_SETUP_PF          0x100

/* PFFFT_DEPTH_FIRST (pffft_pass_config.h) for a new complex setup. The
   backends' setup functions get PFFFT_DF_AUTO only for a buffer that
   reaches the threshold of pffft_pass_config.h for the CPU and lane count
   (pffft_internal_new_setup() turns it into PFFFT_DF_OFF below it). */
#define PFFFT_DF_OFF            (-1) /* never */
#define PFFFT_DF_AUTO           0    /* where the plan allows it, with the leaves of pffft_pass_config.h */
#define PFFFT_DF_ON             1    /* wherever the plan allows it, and as deep as it allows */

/*
  N
  Ncvec      nb of complex simd vectors (N/4 if PFFFT_COMPLEX, N/8 if PFFFT_REAL)
  ifac       the decomposition of N into small integers: N, number of
             factors, factors (admitted values: 2, 3, 4, 5, 6 or 8)
  kind       PFFFT_SETUP_KIND_*: which backend created the setup; a
             native setup may add PFFFT_SETUP_PF (W8 setups never do, so
             kind == PFFFT_SETUP_KIND_W8 tells the backend)
  data       allocated room for twiddle coefs
  e          points into 'data', N/4*3 elements
  twiddle    points into 'data', N/4 elements
  df_levels  PFFFT_DEPTH_FIRST (pffft_passes_impl.h): 0 when all passes
             sweep the buffer, else 1 + the number of depth-first passes
             within a slice; the order of a slice's regions then follows
             the twiddles in 'data'. Where pointers have 8 bytes the
             member takes the place of padding: the size of the structure
             decides where the allocator puts 'data', and that place
             relative to the caller's buffers shows in the timings of
             small transforms.
*/
// The macro arguments are type names in member declarations.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define PFFFT_SETUP_FIELDS(vector_type, real_type)                                                                     \
    int N;                                                                                                             \
    int Ncvec;                                                                                                         \
    int ifac[IFAC_MAX_SIZE];                                                                                           \
    pffft_transform_t transform;                                                                                       \
    int kind;                                                                                                          \
    int df_levels;                                                                                                     \
    vector_type* data;                                                                                                 \
    real_type* e;                                                                                                      \
    real_type* twiddle;

// NOLINTEND(bugprone-macro-parentheses)

#endif /* PFFFT_SETUP_H */
