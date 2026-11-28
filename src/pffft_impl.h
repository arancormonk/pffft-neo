// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/* Modifications: Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com> */
/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB
   (http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
   of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.


   PFFFT : a Pretty Fast FFT.

   This file is largerly based on the original FFTPACK implementation, modified in
   order to take advantage of SIMD instructions of modern CPUs.
*/

/*
  ChangeLog:
  - 2011/10/02, version 1: This is the very first release of this file.
*/

#ifndef PFFFT_IMPL_H
#define PFFFT_IMPL_H

#include "pffft.h"

/* detect compiler flavour */
#if defined(_MSC_VER)
#define COMPILER_MSVC
#elif defined(__GNUC__)
#define COMPILER_GCC
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#if defined(COMPILER_GCC)
#define ALWAYS_INLINE(return_type) inline return_type __attribute__((always_inline))
#define NEVER_INLINE(return_type)  return_type __attribute__((noinline))
#define RESTRICT                   __restrict
/* The public API uses stack scratch when the caller passes a null work buffer. */
#define VLA_ARRAY_ON_STACK(type__, varname__, size__)                                                                  \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wvla\"") type__ varname__[size__];               \
    _Pragma("GCC diagnostic pop")
#elif defined(COMPILER_MSVC)
#define ALWAYS_INLINE(return_type)                    __forceinline return_type
#define NEVER_INLINE(return_type)                     __declspec(noinline) return_type
#define RESTRICT                                      __restrict
#define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__* varname__ = (type__*)_alloca(size__ * sizeof(type__))
#endif

#ifdef COMPILER_MSVC
#pragma warning(disable : 4244 4305 4204 4456)
#endif

/* SIMD operations used by the shared transform code. */
#include "simd/pf_float.h"
#include "pffft_internal.h"
#include "pffft_wide_config.h"

#if defined(PFFFT_WIDE_TU_PART)
/*
  pffft_wide_tu.c: only the 8-lane backend (and nothing that runs before the
  CPU check), compiled with /arch:AVX2. pffft.c's own translation unit has
  everything else.
*/
#if PFFFT_WIDE
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039 /* 1/sqrt(2) */
#endif
/* MSVC's <math.h> defines cosl() and sinl() as inline functions. Copies
   compiled here with /arch:AVX2 could be the ones the linker keeps for
   pffft.c too (it merges identical inline functions), so this file calls
   the out-of-line cos() and sin(): long double is double on MSVC. */
#define cosl(x) cos(x)
#define sinl(x) sin(x)
#include "pffft_setup.h"

struct PFFFT_Setup {
    PFFFT_SETUP_FIELDS(v4sf, float)
};

#include "pffft_wide_impl.h"
#endif /* PFFFT_WIDE */

#elif defined(PFFFT_PREFETCH_TU_PART)
/*
  pffft_prefetch_tu.c: the transforms of the native 4-lane implementation
  again, with passes that prefetch the lines they write (PFFFT_PREFETCH,
  pffft_passes_impl.h), under their own names; the public functions of
  pffft.c call pffft_native_transform_pf and
  pffft_native_transform_ordered_pf for a setup with PFFFT_SETUP_PF. In a
  translation unit of its own, so that pffft.c compiles the code of a build
  without them. Setup, convolution and validation are not compiled here;
  zreorder and the unordered complex finalize and preprocess are pffft.c's.
  Baseline instruction set, as pffft.c.
*/
#if PFFFT_PREFETCH
#define SETUP_STRUCT             PFFFT_Setup
#define FUNC_TRANSFORM_UNORDRD   pffft_native_transform_pf
#define FUNC_TRANSFORM_ORDERED   pffft_native_transform_ordered_pf
#define FUNC_TRANSFORM_INTERNAL  pffft_pf_transform_internal
#define FUNC_ZREORDER            pffft_native_zreorder
#define FUNC_CPLX_FINALIZE       pffft_cplx_finalize
#define FUNC_CPLX_PREPROCESS     pffft_cplx_preprocess
#define FUNC_REAL_PREPROCESS_4X4 pffft_real_preprocess_4x4
#define FUNC_REAL_PREPROCESS     pffft_real_preprocess
#define FUNC_REAL_FINALIZE_4X4   pffft_real_finalize_4x4
#define FUNC_REAL_FINALIZE       pffft_real_finalize
#define FUNC_ALIGNED_MALLOC      pffft_aligned_malloc
#define FUNC_ALIGNED_FREE        pffft_aligned_free
void pffft_native_transform_pf(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                               pffft_direction_t direction);
void pffft_native_transform_ordered_pf(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                                       pffft_direction_t direction);
void pffft_pf_transform_internal(const PFFFT_Setup* setup, const float* in, float* out, v4sf* work,
                                 pffft_direction_t direction, int ordered);
void pffft_native_zreorder(const PFFFT_Setup* setup, const float* input, float* output, pffft_direction_t direction);
void pffft_cplx_finalize(int n, const v4sf* in, v4sf* out, const v4sf* e);
void pffft_cplx_preprocess(int n, const v4sf* in, v4sf* out, const v4sf* e);
/* the setup code, the convolutions and the twiddle tables compile here
   too, unused: static functions of the shared headers */
#if defined(__GNUC__) || defined(__clang__) /* clang-cl too */
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4505) /* unreferenced function with internal linkage */
#endif
#include "pffft_priv_impl.h"
#endif /* PFFFT_PREFETCH */

#else /* PFFFT_WIDE_TU_PART, PFFFT_PREFETCH_TU_PART */

/* have code comparable with this definition */
#define SETUP_STRUCT PFFFT_Setup
#if PFFFT_WIDE
/* the public functions below dispatch to these or to the 8-lane backend */
#define FUNC_NEW_SETUP            pffft_native_new_setup
#define FUNC_DESTROY              pffft_native_destroy_setup
#define FUNC_TRANSFORM_UNORDRD    pffft_native_transform
#define FUNC_TRANSFORM_ORDERED    pffft_native_transform_ordered
#define FUNC_ZREORDER             pffft_native_zreorder
#define FUNC_ZCONVOLVE_ACCUMULATE pffft_native_zconvolve_accumulate
#define FUNC_ZCONVOLVE            pffft_native_zconvolve
#define FUNC_ZCONVERT_ZP          pffft_native_zconvert_zp
#define FUNC_ZCONVOLVE_ZP         pffft_native_zconvolve_zp
#define FUNC_ZCONVOLVE_SCALE      pffft_native_zconvolve_scale
PFFFT_Setup* pffft_native_new_setup(int N, pffft_transform_t transform, int depth_first);
void pffft_native_destroy_setup(PFFFT_Setup* setup);
void pffft_native_transform(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                            pffft_direction_t direction);
void pffft_native_transform_ordered(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                                    pffft_direction_t direction);
void pffft_native_zreorder(const PFFFT_Setup* setup, const float* input, float* output, pffft_direction_t direction);
void pffft_native_zconvolve_accumulate(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab,
                                       float scaling);
void pffft_native_zconvolve_scale(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab,
                                  float scaling);
void pffft_native_zconvolve(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab);
int pffft_native_zconvert_zp(const PFFFT_Setup* setup, const float* in, float* out, float scaling);
int pffft_native_zconvolve_zp(const PFFFT_Setup* setup, const float* dft_x, const float* dft_hzp, float* dft_ab);
#if PFFFT_PREFETCH
/* pffft_prefetch_tu.c */
void pffft_native_transform_pf(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                               pffft_direction_t direction);
void pffft_native_transform_ordered_pf(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                                       pffft_direction_t direction);
#endif
#else
PFFFT_Setup* pffft_native_new_setup(int N, pffft_transform_t transform, int depth_first);
#define FUNC_NEW_SETUP            pffft_native_new_setup
#define FUNC_DESTROY              pffft_destroy_setup
#define FUNC_TRANSFORM_UNORDRD    pffft_transform
#define FUNC_TRANSFORM_ORDERED    pffft_transform_ordered
#define FUNC_ZREORDER             pffft_zreorder
#define FUNC_ZCONVOLVE_ACCUMULATE pffft_zconvolve_accumulate
#define FUNC_ZCONVOLVE            pffft_zconvolve
#define FUNC_ZCONVERT_ZP          pffft_zconvert_zp
#define FUNC_ZCONVOLVE_ZP         pffft_zconvolve_zp
#define FUNC_ZCONVOLVE_SCALE      pffft_zconvolve_scale
#endif

#define FUNC_ALIGNED_MALLOC      pffft_aligned_malloc
#define FUNC_ALIGNED_FREE        pffft_aligned_free
#define FUNC_SIMD_SIZE           pffft_simd_size
#define FUNC_MIN_FFT_SIZE        pffft_min_fft_size
#define FUNC_IS_VALID_SIZE       pffft_is_valid_size
#define FUNC_NEAREST_SIZE        pffft_nearest_transform_size
#define FUNC_SIMD_ARCH           pffft_simd_arch
#define FUNC_VALIDATE_SIMD_A     validate_pffft_simd
#define FUNC_VALIDATE_SIMD_EX    validate_pffft_simd_ex

#define FUNC_CPLX_FINALIZE       pffft_cplx_finalize
#define FUNC_CPLX_PREPROCESS     pffft_cplx_preprocess
#define FUNC_REAL_PREPROCESS_4X4 pffft_real_preprocess_4x4
#define FUNC_REAL_PREPROCESS     pffft_real_preprocess
#define FUNC_REAL_FINALIZE_4X4   pffft_real_finalize_4x4
#define FUNC_REAL_FINALIZE       pffft_real_finalize
#define FUNC_TRANSFORM_INTERNAL  pffft_transform_internal

/* Internal symbols retained by the original single-precision implementation. */
void pffft_transform_internal(const PFFFT_Setup* setup, const float* in, float* out, v4sf* work,
                              pffft_direction_t direction, int ordered);
#if SIMD_SZ == 4
void pffft_cplx_finalize(int n, const v4sf* in, v4sf* out, const v4sf* e);
void pffft_cplx_preprocess(int n, const v4sf* in, v4sf* out, const v4sf* e);
#endif
#include "pffft_priv_impl.h"

int
pffft_setup_simd_size(const PFFFT_Setup* setup) {
    return setup->kind == PFFFT_SETUP_KIND_W8 ? 8 : SIMD_SZ;
}

int
pffft_internal_depth_first_levels(const PFFFT_Setup* setup) {
    return setup->df_levels;
}

const int*
pffft_internal_plan(const PFFFT_Setup* setup, int* count) {
    /* ifac: N, the number of passes, their radices */
    *count = setup->ifac[1];
    return setup->ifac + 2;
}

/* the PFFFT_DF_* value of the PFFFT_BACKEND_* flags */
static int
pffft_backend_depth_first(int backend) {
    return (backend & PFFFT_BACKEND_DEPTH_FIRST)     ? PFFFT_DF_ON
           : (backend & PFFFT_BACKEND_BREADTH_FIRST) ? PFFFT_DF_OFF
                                                     : PFFFT_DF_AUTO;
}

/* the flags added to a PFFFT_BACKEND_* backend */
#define PFFFT_BACKEND_MODIFIERS                                                                                        \
    (PFFFT_BACKEND_DEPTH_FIRST | PFFFT_BACKEND_BREADTH_FIRST | PFFFT_BACKEND_PREFETCH | PFFFT_BACKEND_NO_PREFETCH)

/* the prefetch mode of the PFFFT_BACKEND_* flags: 1 forced on, -1 off, 0
   as pffft_new_setup() picks it */
static int
pffft_backend_prefetch(int backend) {
    return (backend & PFFFT_BACKEND_PREFETCH) ? 1 : (backend & PFFFT_BACKEND_NO_PREFETCH) ? -1 : 0;
}

int
pffft_internal_prefetch_passes(const PFFFT_Setup* setup) {
    return (setup->kind & PFFFT_SETUP_PF) != 0;
}

#if defined(__GNUC__)
#define PFFFT_UNLIKELY(c) __builtin_expect((c) != 0, 0)
#else
#define PFFFT_UNLIKELY(c) (c)
#endif

#if PFFFT_WIDE

#include "pffft_wide_select.h"
#if defined(PFFFT_WIDE_SEPARATE)
#include "pffft_wide_api.h"
#else
#include "pffft_wide_impl.h"
#endif
/* <intrin.h> for MSVC and clang-cl, whose <cpuid.h> __cpuid macro clashes with it */
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static void
pffft_x86_cpuid(void* ctx, unsigned leaf, unsigned subleaf, unsigned* r) {
    (void)ctx;
#if defined(_MSC_VER)
    {
        int v[4];
        __cpuidex(v, (int)leaf, (int)subleaf);
        r[0] = (unsigned)v[0];
        r[1] = (unsigned)v[1];
        r[2] = (unsigned)v[2];
        r[3] = (unsigned)v[3];
    }
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    r[0] = a;
    r[1] = b;
    r[2] = c;
    r[3] = d;
#endif
}

/* a setup of the native implementation, with PFFFT_SETUP_PF as
   'prefetch' (pffft_backend_prefetch()) asks */
static PFFFT_Setup*
pffft_native_setup(int N, pffft_transform_t transform, int depth_first, int prefetch) {
    PFFFT_Setup* s = pffft_native_new_setup(N, transform, depth_first);
#if PFFFT_PREFETCH
    /* depth-first setups run other passes (pffft_passes_impl.h) */
    if (s && prefetch >= 0 && !s->df_levels) {
        if (prefetch > 0) {
            s->kind |= PFFFT_SETUP_PF;
        } else {
            pffft_x86_cpu_ops ops;
            ops.cpuid = pffft_x86_cpuid;
            ops.xgetbv = NULL; /* not needed */
            ops.ctx = NULL;
            if (pffft_x86_prefetch_passes(&ops, 2 * (unsigned long long)s->Ncvec * sizeof(v4sf),
                                          transform == PFFFT_REAL)) {
                s->kind |= PFFFT_SETUP_PF;
            }
        }
    }
#else
    (void)prefetch;
#endif
    return s;
}

static unsigned long long
pffft_x86_xgetbv(void* ctx, unsigned index) {
    (void)ctx;
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(index);
#else
    {
        unsigned lo, hi;
        /* xgetbv, spelled out for assemblers that do not know it */
        // NOLINTNEXTLINE(portability-no-assembler) -- baseline dispatch reads XCR0 without enabling AVX.
        __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(index));
        return ((unsigned long long)hi << 32) | lo;
    }
#endif
}

#if defined(PFFFT_DISABLE_FMA)
#define PFFFT_W8_NEED_FMA 0
#else
#define PFFFT_W8_NEED_FMA 1
#endif

/* -1 if the CPU cannot run the 8-lane backend, else its PFFFT_X86_WIDE_*
   class with the build options applied; *l2 receives the L2 size the
   inferred class needs (0 otherwise or when unknown) */
static int
pffft_wide_cpu_class(unsigned long long* l2) {
    pffft_x86_cpu_ops ops;
    pffft_x86_cpu_id id;
    int cls;
    ops.cpuid = pffft_x86_cpuid;
    ops.xgetbv = pffft_x86_xgetbv;
    ops.ctx = NULL;
    *l2 = 0;
    if (!pffft_x86_probe_wide(&ops, PFFFT_W8_NEED_FMA, &id)) {
        return -1;
    }
#if defined(PFFFT_WIDE_UNKNOWN_CPUS)
    (void)id;
    cls = PFFFT_X86_WIDE_ZEN3; /* its table for every AVX2 CPU */
#else
    cls = pffft_x86_wide_class(&id);
#if !defined(PFFFT_WIDE_INFERRED)
    /* not measured: 4 lanes unless the build opts in */
    if (cls == PFFFT_X86_WIDE_INFERRED) {
        cls = PFFFT_X86_WIDE_OFF;
    }
#endif
    if (cls == PFFFT_X86_WIDE_INFERRED) {
        *l2 = pffft_x86_l2_bytes(&ops, &id);
    }
#endif
    return cls;
}

PFFFT_Setup*
pffft_internal_new_setup(int N, pffft_transform_t transform, int backend) {
    const int depth_first = pffft_backend_depth_first(backend), prefetch = pffft_backend_prefetch(backend);
    backend &= ~PFFFT_BACKEND_MODIFIERS;
    if (backend != PFFFT_BACKEND_NATIVE) {
        unsigned long long l2 = 0;
        const int cls = pffft_wide_supports(N, transform) ? pffft_wide_cpu_class(&l2) : -1;
        if (backend == PFFFT_BACKEND_W8) {
            return cls >= 0 ? pffft_w8_new_setup(N, transform, depth_first) : NULL;
        }
        /* an allocation failure is reported, not retried with the other backend */
#if defined(PFFFT_WIDE_ALL_SIZES)
        /* benchmarking: every supported size on every capable CPU */
        if (cls >= 0) {
            return pffft_w8_new_setup(N, transform, depth_first);
        }
#endif
        if (cls >= 0 && pffft_wide_default(N, transform, cls, l2)) {
            return pffft_w8_new_setup(N, transform, depth_first);
        }
    }
    return pffft_native_setup(N, transform, depth_first, prefetch);
}

void
pffft_destroy_setup(PFFFT_Setup* setup) {
    /* both kinds hold the same two allocations */
    pffft_native_destroy_setup(setup);
}

void
pffft_transform(const PFFFT_Setup* setup, const float* input, float* output, float* work, pffft_direction_t direction) {
#if PFFFT_PREFETCH
    /* one comparison for a native setup without prefetches, as without
       PFFFT_PREFETCH; with the other setups out of the way (without the
       hint, GCC compiled the native path with another prologue, and
       transforms of 16 to 64 points took 2 to 7% longer on Zen 3) */
    if (PFFFT_UNLIKELY(setup->kind != PFFFT_SETUP_KIND_NATIVE)) {
        if (setup->kind == PFFFT_SETUP_KIND_W8) {
            pffft_w8_transform(setup, input, output, work, direction);
        } else {
            pffft_native_transform_pf(setup, input, output, work, direction); /* PFFFT_SETUP_PF */
        }
    } else {
        pffft_native_transform(setup, input, output, work, direction);
    }
#else
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_transform(setup, input, output, work, direction);
    } else {
        pffft_native_transform(setup, input, output, work, direction);
    }
#endif
}

void
pffft_transform_ordered(const PFFFT_Setup* setup, const float* input, float* output, float* work,
                        pffft_direction_t direction) {
#if PFFFT_PREFETCH
    /* one comparison for a native setup without prefetches, as without
       PFFFT_PREFETCH; with the other setups out of the way (without the
       hint, GCC compiled the native path with another prologue, and
       transforms of 16 to 64 points took 2 to 7% longer on Zen 3) */
    if (PFFFT_UNLIKELY(setup->kind != PFFFT_SETUP_KIND_NATIVE)) {
        if (setup->kind == PFFFT_SETUP_KIND_W8) {
            pffft_w8_transform_ordered(setup, input, output, work, direction);
        } else {
            pffft_native_transform_ordered_pf(setup, input, output, work, direction); /* PFFFT_SETUP_PF */
        }
    } else {
        pffft_native_transform_ordered(setup, input, output, work, direction);
    }
#else
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_transform_ordered(setup, input, output, work, direction);
    } else {
        pffft_native_transform_ordered(setup, input, output, work, direction);
    }
#endif
}

void
pffft_zreorder(const PFFFT_Setup* setup, const float* input, float* output, pffft_direction_t direction) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_zreorder(setup, input, output, direction);
    } else {
        pffft_native_zreorder(setup, input, output, direction);
    }
}

void
pffft_zconvolve_accumulate(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab,
                           float scaling) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_zconvolve_accumulate(setup, dft_a, dft_b, dft_ab, scaling);
    } else {
        pffft_native_zconvolve_accumulate(setup, dft_a, dft_b, dft_ab, scaling);
    }
}

void
pffft_zconvolve_scale(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab, float scaling) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
    } else {
        pffft_native_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
    }
}

void
pffft_zconvolve(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        pffft_w8_zconvolve(setup, dft_a, dft_b, dft_ab);
    } else {
        pffft_native_zconvolve(setup, dft_a, dft_b, dft_ab);
    }
}

int
pffft_zconvert_zp(const PFFFT_Setup* setup, const float* in, float* out, float scaling) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        return pffft_w8_zconvert_zp(setup, in, out, scaling);
    }
    return pffft_native_zconvert_zp(setup, in, out, scaling);
}

int
pffft_zconvolve_zp(const PFFFT_Setup* setup, const float* dft_x, const float* dft_hzp, float* dft_ab) {
    if (setup->kind == PFFFT_SETUP_KIND_W8) {
        return pffft_w8_zconvolve_zp(setup, dft_x, dft_hzp, dft_ab);
    }
    return pffft_native_zconvolve_zp(setup, dft_x, dft_hzp, dft_ab);
}

#else /* PFFFT_WIDE */

/* without the 8-lane backend's CPU probe no prefetching passes either
   (PFFFT_PREFETCH) */
static PFFFT_Setup*
pffft_native_setup(int N, pffft_transform_t transform, int depth_first, int prefetch) {
    (void)prefetch;
    return pffft_native_new_setup(N, transform, depth_first);
}

PFFFT_Setup*
pffft_internal_new_setup(int N, pffft_transform_t transform, int backend) {
    const int depth_first = pffft_backend_depth_first(backend), prefetch = pffft_backend_prefetch(backend);
    backend &= ~PFFFT_BACKEND_MODIFIERS;
    if (backend == PFFFT_BACKEND_W8) {
        return NULL;
    }
    return pffft_native_setup(N, transform, depth_first, prefetch);
}

#endif /* PFFFT_WIDE */

PFFFT_Setup*
pffft_new_setup(int N, pffft_transform_t transform) {
    return pffft_internal_new_setup(N, transform, PFFFT_BACKEND_AUTO);
}

/* deprecated synonym for pffft_zconvolve_scale() -- kept for ABI compatibility */
void
pffft_zconvolve_no_accu(const PFFFT_Setup* setup, const float* dft_a, const float* dft_b, float* dft_ab,
                        float scaling) {
    pffft_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
}

#endif /* PFFFT_WIDE_TU_PART */

#endif /* PFFFT_IMPL_H */
