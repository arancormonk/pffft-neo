// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Accuracy and contract test for pffft.

  Every valid transform size up to a limit is checked against an
  independent reference DFT computed in long double (fft_reference.h):

    forward:  X[k] = sum_n x[n] exp(-2*pi*i*n*k/N)
    backward: x[n] = sum_k X[k] exp(+2*pi*i*n*k/N)   (unnormalized)

  for PFFFT_REAL and PFFFT_COMPLEX, ordered and unordered, together with
  the identities the API documents: zreorder() round trips, in-place
  operation, work == NULL, the zconvolve family as circular convolution,
  the zero-phase path, and the size helpers. Complex sizes whose plan
  allows the depth-first pass schedule (PFFFT_DEPTH_FIRST) are checked
  again with it, and it must give the forward results of the other
  schedule bit for bit.

  Error criterion: the relative L2 error of a backward-stable FFT is
  O(eps * log2 N); a defect (wrong twiddle, layout, sign, scaling) shows
  up as an O(1) error.  Sizes fail above TOL_FFT * eps * log2 N.

  usage: test_pffft_accuracy_float [max_N]
*/

#include "pffft/pffft.h"
#include "pffft_internal.h"
#include "pffft_setup.h"
#include "fft_reference.h"
/* the pass schedule switches of the build for the depth-first checks:
   those of the 4-lane backends and of the 8-lane one, which is built next
   to the 4-lane SSE one. pffft_pass_config.h wants a SIMD_SZ, which a
   definition of the build may use, so they are read while it is defined. */
#define SIMD_SZ 4
#include "pffft_pass_config.h"
static const int t_depth_first = PFFFT_DEPTH_FIRST;
static const int t_terminal_radix8 = PFFFT_TERMINAL_RADIX8;
static const unsigned long long t_df_min_bytes_wide = (unsigned long long)(PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE);
#undef SIMD_SZ
#include "test_size_arg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

/* accepted relative L2 error, in units of eps * log2(N) */
#define TOL_FFT               2.0
/* default largest N; the suite stays within a few seconds */
#define DEFAULT_MAX_N         8192
/* sizes up to this get the impulse-at-every-position check */
#define IMPULSE_MAX_N         64
#define MAX_REPORTED_FAILURES 50

static int num_failures = 0;

static void
report_failure(const char* what, int N, int cplx, double err) {
    ++num_failures;
    if (num_failures <= MAX_REPORTED_FAILURES) {
        printf("FAIL: %s %s N=%d rel_err=%g (%.2f eps*log2N)\n", what, cplx ? "COMPLEX" : "REAL", N, err,
               err / (FLT_EPSILON * (N > 1 ? log2((double)N) : 1.0)));
    }
}

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;

static double
urand(void) {
    return xorshift_urand(&rng_state);
}

/* circular convolution a (*) b via the reference DFT */
static void
ref_cconv(int N, const cld* a, const cld* b, cld* out) {
    cld *A = alloc_reference(N), *B = alloc_reference(N);
    int k;
    ref_dft(N, a, A, -1);
    ref_dft(N, b, B, -1);
    for (k = 0; k < N; ++k) {
        A[k] = cmul(A[k], B[k]);
    }
    ref_dft(N, A, out, +1);
    for (k = 0; k < N; ++k) {
        out[k].re /= N;
        out[k].im /= N;
    }
    free(A);
    free(B);
}

/* ---------------- helpers ---------------- */

static double
log2N(int N) {
    return N > 1 ? log2((double)N) : 1.0;
}

/* relative L2 error; y holds n complex values interleaved (cplx) or n reals */
static double
rel_err(int n, const float* y, const cld* ref, int cplx) {
    ldouble num = 0, den = 0;
    add_err_sums(n, y, ref, cplx, &num, &den);
    return (double)(den > 0 ? sqrtl(num / den) : sqrtl(num));
}

static void
scale_add(int N, cld* acc, const cld* v, ldouble s) {
    int k;
    for (k = 0; k < N; ++k) {
        acc[k].re += s * v[k].re;
        acc[k].im += s * v[k].im;
    }
}

static int
same_bits(const float* a, const float* b, int n) {
    return memcmp(a, b, sizeof(float) * n) == 0;
}

static float*
alloc_buf(int n) {
    float* buffer = pffft_aligned_malloc(sizeof(float) * (size_t)n);
    if (!buffer) {
        fputs("Could not allocate FFT test buffer\n", stderr);
        exit(EXIT_FAILURE);
    }
    return buffer;
}

static int
is_smooth(int m) {
    while (m % 2 == 0) {
        m /= 2;
    }
    while (m % 3 == 0) {
        m /= 3;
    }
    while (m % 5 == 0) {
        m /= 5;
    }
    return m == 1;
}

/* ---------------- per-size checks ---------------- */

typedef void (*transform_fn)(const PFFFT_Setup*, const float*, float*, float*, pffft_direction_t);

static double worst_fwd = 0, worst_bwd = 0, worst_conv = 0;

static void
check_err(const char* what, int N, int cplx, double err, double tol_mult, double* worst) {
    double norm = err / (FLT_EPSILON * log2N(N));
    if (worst && norm > *worst) {
        *worst = norm;
    }
    if (!(norm <= tol_mult * TOL_FFT)) {
        report_failure(what, N, cplx, err);
    }
}

/* which implementation test_size() exercises (PFFFT_BACKEND_*) */
static int g_backend = PFFFT_BACKEND_AUTO;

static PFFFT_Setup*
new_test_setup(int N, int cplx) {
    pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
    PFFFT_Setup* s = pffft_internal_new_setup(N, tt, g_backend);
    if (s) {
        const int backend = g_backend & ~(PFFFT_BACKEND_DEPTH_FIRST | PFFFT_BACKEND_BREADTH_FIRST);
        const int lanes = pffft_setup_simd_size(s);
        if ((backend == PFFFT_BACKEND_NATIVE && lanes != pffft_simd_size())
            || (backend == PFFFT_BACKEND_W8 && lanes != 8)) {
            report_failure("setup uses an unexpected backend", N, cplx, (double)lanes);
        }
        if ((g_backend & PFFFT_BACKEND_BREADTH_FIRST) && pffft_internal_depth_first_levels(s)) {
            report_failure("breadth-first setup is depth first", N, cplx, 1);
        }
    }
    return s;
}

static void
test_size(int N, int cplx) {
    if (N <= 0 || N > PFFFT_MAX_N || (!cplx && N % 2 != 0)) {
        report_failure("invalid reference FFT size", N, cplx, 1);
        return;
    }
    PFFFT_Setup* s = new_test_setup(N, cplx);
    const int nf = cplx ? 2 * N : N; /* scalars per buffer */
    const int ns = cplx ? N : N / 2; /* complex slots in the ordered layout */
    float *x = alloc_buf(nf), *u = alloc_buf(nf), *y = alloc_buf(nf), *z = alloc_buf(nf);
    float *w = alloc_buf(nf), *work = alloc_buf(nf);
    float *a = alloc_buf(nf), *b = alloc_buf(nf), *d = alloc_buf(nf);
    float *A = alloc_buf(nf), *B = alloc_buf(nf);
    cld *xc = alloc_reference(N), *Fc = alloc_reference(N);
    cld* ref = alloc_reference(N);
    cld *ac = alloc_reference(N), *bc = alloc_reference(N);
    cld *dc = alloc_reference(N), *cv = alloc_reference(N);
    int k, j, ord, dir, variant;

    if (!s) {
        report_failure("new_setup() returned NULL for a valid size", N, cplx, 1.0);
        goto done;
    }

    for (k = 0; k < nf; ++k) {
        x[k] = (float)urand();
        u[k] = (float)urand();
        a[k] = (float)urand();
        b[k] = (float)urand();
        d[k] = (float)urand();
    }

    /* forward, ordered */
    to_cld(N, x, cplx, xc);
    ref_dft(N, xc, Fc, -1);
    pffft_transform_ordered(s, x, y, work, PFFFT_FORWARD);
    if (cplx) {
        check_err("forward", N, cplx, rel_err(N, y, Fc, 1), 1, &worst_fwd);
    } else {
        real_spectrum_to_ordered(N, Fc, ref);
        check_err("forward", N, cplx, rel_err(ns, y, ref, 1), 1, &worst_fwd);
    }

    /* backward, ordered, from an arbitrary spectrum */
    if (cplx) {
        to_cld(N, u, 1, Fc);
    } else {
        ordered_to_real_spectrum(N, u, Fc);
    }
    ref_dft(N, Fc, ref, +1);
    pffft_transform_ordered(s, u, y, work, PFFFT_BACKWARD);
    check_err("backward", N, cplx, rel_err(N, y, ref, cplx), 1, &worst_bwd);

    /* unordered + zreorder must reproduce the ordered transforms bit for bit */
    pffft_transform_ordered(s, x, y, work, PFFFT_FORWARD);
    pffft_transform(s, x, z, work, PFFFT_FORWARD);
    pffft_zreorder(s, z, w, PFFFT_FORWARD);
    if (!same_bits(y, w, nf)) {
        report_failure("zreorder(transform) != transform_ordered (forward)", N, cplx, 1);
    }
    pffft_zreorder(s, w, A, PFFFT_BACKWARD);
    if (!same_bits(A, z, nf)) {
        report_failure("zreorder backward does not invert zreorder forward", N, cplx, 1);
    }
    pffft_transform_ordered(s, u, y, work, PFFFT_BACKWARD);
    pffft_zreorder(s, u, w, PFFFT_BACKWARD);
    pffft_transform(s, w, z, work, PFFFT_BACKWARD);
    if (!same_bits(y, z, nf)) {
        report_failure("transform(zreorder) != transform_ordered (backward)", N, cplx, 1);
    }

    /* in-place, work == NULL, garbage in work, input left untouched */
    for (ord = 0; ord < 2; ++ord) {
        for (dir = 0; dir < 2; ++dir) {
            transform_fn fn = ord ? pffft_transform_ordered : pffft_transform;
            pffft_direction_t direction = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
            const float* src = dir ? u : x;
            fn(s, src, y, work, direction);
            memcpy(z, src, sizeof(float) * nf);
            fn(s, z, z, work, direction);
            if (!same_bits(y, z, nf)) {
                report_failure(ord ? "in-place transform_ordered" : "in-place transform", N, cplx, 1);
            }
            memcpy(z, src, sizeof(float) * nf);
            fn(s, z, z, NULL, direction);
            if (!same_bits(y, z, nf)) {
                report_failure("in-place with work == NULL", N, cplx, 1);
            }
            fn(s, src, z, NULL, direction);
            if (!same_bits(y, z, nf)) {
                report_failure("work == NULL", N, cplx, 1);
            }
            for (k = 0; k < nf; ++k) {
                work[k] = (float)1e30;
            }
            memcpy(w, src, sizeof(float) * nf);
            fn(s, w, z, work, direction);
            if (!same_bits(y, z, nf)) {
                report_failure("result depends on work contents", N, cplx, 1);
            }
            if (!same_bits(w, src, nf)) {
                report_failure("out-of-place transform modified its input", N, cplx, 1);
            }
        }
    }

    /* backward(forward(x)) == N * x */
    pffft_transform(s, x, z, work, PFFFT_FORWARD);
    pffft_transform(s, z, y, work, PFFFT_BACKWARD);
    to_cld(N, x, cplx, ref);
    for (k = 0; k < N; ++k) {
        ref[k].re *= N;
        ref[k].im *= N;
    }
    check_err("round trip", N, cplx, rel_err(N, y, ref, cplx), 2, NULL);

    /* zconvolve family: backward(D + s*A*B) == N*d + s*N*(a (*) b) */
    to_cld(N, a, cplx, ac);
    to_cld(N, b, cplx, bc);
    to_cld(N, d, cplx, dc);
    ref_cconv(N, ac, bc, cv);
    pffft_transform(s, a, A, work, PFFFT_FORWARD);
    pffft_transform(s, b, B, work, PFFFT_FORWARD);
    for (variant = 0; variant < 7; ++variant) {
        const float sc = (float)0.75;
        const cld* base = NULL; /* signal whose spectrum is accumulated onto */
        const char* what;
        ldouble prod_scale = 1;
        switch (variant) {
            case 0:
                what = "zconvolve_scale";
                pffft_zconvolve_scale(s, A, B, w, sc);
                prod_scale = sc;
                break;
            case 1:
                what = "zconvolve";
                pffft_zconvolve(s, A, B, w);
                break;
            case 2:
                what = "zconvolve_accumulate";
                pffft_transform(s, d, w, work, PFFFT_FORWARD);
                pffft_zconvolve_accumulate(s, A, B, w, sc);
                prod_scale = sc;
                base = dc;
                break;
            case 3:
                what = "zconvolve_scale, ab == a";
                memcpy(w, A, sizeof(float) * nf);
                pffft_zconvolve_scale(s, w, B, w, sc);
                prod_scale = sc;
                break;
            case 4:
                what = "zconvolve_accumulate, ab == b";
                memcpy(w, B, sizeof(float) * nf);
                pffft_zconvolve_accumulate(s, A, w, w, sc);
                prod_scale = sc;
                base = bc;
                break;
            case 5:
                what = "zconvolve, ab == a";
                memcpy(w, A, sizeof(float) * nf);
                pffft_zconvolve(s, w, B, w);
                break;
            default:
                what = "zconvolve_scale, a == b == ab";
                memcpy(w, A, sizeof(float) * nf);
                pffft_zconvolve_scale(s, w, w, w, sc);
                prod_scale = sc;
                break;
        }
        pffft_transform(s, w, y, work, PFFFT_BACKWARD);
        memset(ref, 0, sizeof(cld) * N);
        if (base) {
            scale_add(N, ref, base, (ldouble)N);
        }
        if (variant == 6) {
            cld* sq = alloc_reference(N);
            ref_cconv(N, ac, ac, sq);
            scale_add(N, ref, sq, prod_scale * N);
            free(sq);
        } else {
            scale_add(N, ref, cv, prod_scale * N);
        }
        check_err(what, N, cplx, rel_err(N, y, ref, cplx), 3, &worst_conv);
    }

    /* zero-phase path: filter h = b reduced to its even part
     h_e[n] = (h[n] + h[-n]) / 2, applied with gain g */
    if (!cplx) {
        const float g = (float)1.5;
        int inplace;
        cld* he = alloc_reference(N);
        for (k = 0; k < N; ++k) {
            he[k].re = ((ldouble)b[k] + (ldouble)b[(N - k) % N]) / 2;
            he[k].im = 0;
        }
        ref_cconv(N, ac, he, ref);
        for (k = 0; k < N; ++k) {
            ref[k].re *= g;
            ref[k].im = 0;
        }
        for (inplace = 0; inplace < 2; ++inplace) {
            int r1, r2;
            pffft_transform(s, b, B, work, PFFFT_FORWARD);
            pffft_transform(s, a, A, work, PFFFT_FORWARD);
            if (inplace) {
                r1 = pffft_zconvert_zp(s, B, B, g);
                r2 = pffft_zconvolve_zp(s, A, B, A);
                memcpy(w, A, sizeof(float) * nf);
            } else {
                r1 = pffft_zconvert_zp(s, B, z, g);
                r2 = pffft_zconvolve_zp(s, A, z, w);
            }
            if (r1 || r2) {
                report_failure("zero-phase functions rejected a REAL setup", N, cplx, 1);
            }
            pffft_transform(s, w, y, work, PFFFT_BACKWARD);
            check_err(inplace ? "zero-phase, in-place" : "zero-phase", N, cplx, rel_err(N, y, ref, 0), 3, &worst_conv);
        }
        free(he);
    } else {
        if (pffft_zconvert_zp(s, A, w, 1) == 0 || pffft_zconvolve_zp(s, A, B, w) == 0) {
            report_failure("zero-phase functions accepted a COMPLEX setup", N, cplx, 1);
        }
    }

    /* impulses: every output depends on exactly one input */
    if (N <= IMPULSE_MAX_N) {
        for (j = 0; j < N; ++j) {
            memset(x, 0, sizeof(float) * nf);
            x[cplx ? 2 * j : j] = 1;
            to_cld(N, x, cplx, xc);
            ref_dft(N, xc, Fc, -1);
            pffft_transform_ordered(s, x, y, work, PFFFT_FORWARD);
            if (cplx) {
                check_err("impulse forward", N, cplx, rel_err(N, y, Fc, 1), 1, NULL);
            } else {
                real_spectrum_to_ordered(N, Fc, ref);
                check_err("impulse forward", N, cplx, rel_err(ns, y, ref, 1), 1, NULL);
            }
        }
    }

done:
    if (s) {
        pffft_destroy_setup(s);
    }
    pffft_aligned_free(x);
    pffft_aligned_free(u);
    pffft_aligned_free(y);
    pffft_aligned_free(z);
    pffft_aligned_free(w);
    pffft_aligned_free(work);
    pffft_aligned_free(a);
    pffft_aligned_free(b);
    pffft_aligned_free(d);
    pffft_aligned_free(A);
    pffft_aligned_free(B);
    free(xc);
    free(Fc);
    free(ref);
    free(ac);
    free(bc);
    free(dc);
    free(cv);
}

/* ---------------- size helpers ---------------- */

static void
test_size_helpers(void) {
    int cplx, N;
    for (cplx = 0; cplx < 2; ++cplx) {
        pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
        const int simd = pffft_simd_size();
        const int nmin = pffft_min_fft_size(tt);
        if (nmin != (cplx ? simd * simd : 2 * simd * simd)) {
            report_failure("min_fft_size", nmin, cplx, 1);
        }
        /* Once the SIMD lane factor is removed, only factors of 2, 3 and 5 are allowed. */
        for (N = -3; N <= 20000; ++N) {
            const int valid = N > 0 && N % nmin == 0 && is_smooth(N / nmin);
            if (pffft_is_valid_size(N, tt) != valid) {
                report_failure("is_valid_size", N, cplx, 1);
            }
            if (N <= 4096) {
                PFFFT_Setup* s = pffft_new_setup(N, tt);
                if ((s != NULL) != valid) {
                    report_failure(s ? "new_setup accepted an invalid size" : "new_setup rejected a valid size", N,
                                   cplx, 1);
                }
                if (s) {
                    pffft_destroy_setup(s);
                }
            }
            if (N >= 1 && N <= 4096) {
                /* Search both ways one size at a time for an independent nearest-size reference. */
                int hi = N, lo = N;
                while (!(hi % nmin == 0 && is_smooth(hi / nmin))) {
                    ++hi;
                }
                while (lo > nmin && !(lo % nmin == 0 && is_smooth(lo / nmin))) {
                    --lo;
                }
                if (lo < nmin) {
                    lo = nmin;
                }
                if (pffft_nearest_transform_size(N, tt, 1) != hi) {
                    report_failure("nearest_transform_size(higher)", N, cplx, 1);
                }
                if (pffft_nearest_transform_size(N, tt, 0) != lo) {
                    report_failure("nearest_transform_size(lower)", N, cplx, 1);
                }
            }
        }
    }
    /* the helpers agree with new_setup() above its 2^26 limit, and invalid
     arguments make new_setup() / aligned_malloc() fail without asserting */
    for (cplx = 0; cplx < 2; ++cplx) {
        pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
        static const int too_big[] = {(1 << 26) + (1 << 20), 3 << 25, 1 << 27, 1 << 30, INT_MAX};
        int i;
        if (!pffft_is_valid_size(1 << 26, tt)) {
            report_failure("is_valid_size(2^26)", 1 << 26, cplx, 1);
        }
        for (i = 0; i < (int)(sizeof(too_big) / sizeof(too_big[0])); ++i) {
            PFFFT_Setup* s = pffft_new_setup(too_big[i], tt);
            if (pffft_is_valid_size(too_big[i], tt)) {
                report_failure("is_valid_size accepted N > 2^26", too_big[i], cplx, 1);
            }
            if (s) {
                report_failure("new_setup accepted N > 2^26", too_big[i], cplx, 1);
                pffft_destroy_setup(s);
            }
            if (pffft_nearest_transform_size(too_big[i], tt, 1) != (1 << 26)) {
                report_failure("nearest_transform_size(higher) above 2^26", too_big[i], cplx, 1);
            }
            if (pffft_nearest_transform_size(too_big[i], tt, 0) != (1 << 26)) {
                report_failure("nearest_transform_size(lower) above 2^26", too_big[i], cplx, 1);
            }
        }
        if (pffft_nearest_transform_size((1 << 26) - 1, tt, 1) != (1 << 26)) {
            report_failure("nearest_transform_size(higher) below 2^26", (1 << 26) - 1, cplx, 1);
        }
    }
    for (N = -1; N <= 64; ++N) {
        PFFFT_Setup* s = pffft_new_setup(N, (pffft_transform_t)2);
        if (s) {
            report_failure("new_setup accepted an invalid transform type", N, 0, 1);
            pffft_destroy_setup(s);
        }
    }
    {
        void* p = pffft_aligned_malloc(SIZE_MAX - 8);
        if (p) {
            report_failure("aligned_malloc(SIZE_MAX - 8) did not return NULL", 0, 0, 1);
            pffft_aligned_free(p);
        }
    }

    if (pffft_is_power_of_two(INT_MIN)) {
        report_failure("is_power_of_two", INT_MIN, 0, 1);
    }
    /* Repeated doubling gives the expected ceiling, including exact powers of two. */
    for (N = 1; N <= (1 << 20); ++N) {
        int p = 1;
        while (p < N) {
            p <<= 1;
        }
        if (pffft_next_power_of_two(N) != p) {
            report_failure("next_power_of_two", N, 0, 1);
            break;
        }
        if (pffft_is_power_of_two(N) != (p == N)) {
            report_failure("is_power_of_two", N, 0, 1);
            break;
        }
    }
}

/* The depth-first schedule of the complex transform of N on 'backend'
   (PFFFT_BACKEND_NATIVE or _W8): forced, with the smallest leaves, or as
   pffft_new_setup() schedules the size, which is depth first from a size
   of pffft_pass_config.h on. The checks of test_size(), and the forward
   spectra must have the bits of the breadth-first schedule's. Returns 1
   if the setup is depth first, else 0 without a check. */
static int
test_depth_first_size(int backend, int N, int forced) {
    const int flags = backend | (forced ? PFFFT_BACKEND_DEPTH_FIRST : 0);
    PFFFT_Setup* df = pffft_internal_new_setup(N, PFFFT_COMPLEX, flags);
    PFFFT_Setup* bf;
    if (!df || !pffft_internal_depth_first_levels(df)) {
        pffft_destroy_setup(df);
        return 0;
    }
    bf = pffft_internal_new_setup(N, PFFFT_COMPLEX, backend | PFFFT_BACKEND_BREADTH_FIRST);
    if (!bf || pffft_internal_depth_first_levels(bf)) {
        report_failure("no breadth-first setup for a depth-first size", N, 1, 1);
    } else {
        float *x = alloc_buf(2 * N), *work = alloc_buf(2 * N);
        float *y = alloc_buf(2 * N), *z = alloc_buf(2 * N);
        int k, ord;
        for (k = 0; k < 2 * N; ++k) {
            x[k] = (float)urand();
        }
        for (ord = 0; ord < 2; ++ord) {
            transform_fn fn = ord ? pffft_transform_ordered : pffft_transform;
            fn(df, x, y, work, PFFFT_FORWARD);
            fn(bf, x, z, work, PFFFT_FORWARD);
            if (!same_bits(y, z, 2 * N)) {
                report_failure(ord ? "depth-first transform_ordered != breadth-first (forward)"
                                   : "depth-first transform != breadth-first (forward)",
                               N, 1, 1);
            }
        }
        pffft_aligned_free(x);
        pffft_aligned_free(work);
        pffft_aligned_free(y);
        pffft_aligned_free(z);
    }
    pffft_destroy_setup(bf);
    pffft_destroy_setup(df);
    g_backend = flags;
    test_size(N, 1);
    g_backend = PFFFT_BACKEND_AUTO;
    return 1;
}

/* whether the complex plan of N on 'backend' allows the depth-first
   schedule: its first 'top' passes (1 with 4 lanes, 2 with 8) radix 4,
   and two more after them */
static int
depth_first_plan(int backend, int N) {
    PFFFT_Setup* s = pffft_internal_new_setup(N, PFFFT_COMPLEX, backend | PFFFT_BACKEND_BREADTH_FIRST);
    int count = 0, top, k, ok;
    const int* f;
    if (!s) {
        return 0;
    }
    f = pffft_internal_plan(s, &count);
    top = pffft_setup_simd_size(s) == 8 ? 2 : 1;
    ok = count >= top + 2;
    for (k = 0; ok && k < top; ++k) {
        ok = f[k] == 4;
    }
    pffft_destroy_setup(s);
    return ok;
}

/* every complex size up to max_n whose plan allows the depth-first
   schedule, forced; returns their number. Where the build has the
   schedule (PFFFT_DEPTH_FIRST, on a backend with 4 or 8 lanes), every
   plan that allows it must take it. */
static int
test_depth_first(int backend, int nmin, int max_n) {
    int N, tested = 0, missed = 0, first_missed = 0;
    for (N = nmin; N <= max_n; N += nmin) {
        if (is_smooth(N / nmin)) {
            const int t = test_depth_first_size(backend, N, 1);
            tested += t;
            if (!t && depth_first_plan(backend, N)) {
                if (!missed) {
                    first_missed = N;
                }
                ++missed;
            }
        }
    }
    if (t_depth_first && (backend == PFFFT_BACKEND_W8 || pffft_simd_size() >= 4) && missed) {
        printf("FAIL: %d COMPLEX plans that allow the depth-first schedule did not take it, the first N=%d\n", missed,
               first_missed);
        ++num_failures;
    }
    return tested;
}

/* The passes that prefetch the lines they write (PFFFT_BACKEND_PREFETCH,
   pffft_prefetch_tu.c) against those that do not: every transform mode,
   out of place and in place, with scratch on the stack (work NULL) or not,
   must give the same bits. Returns 1 if the build has them, else 0 without
   a check. */
static int
test_prefetch_size(int N, int cplx) {
    const pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
    const int bf = PFFFT_BACKEND_NATIVE | PFFFT_BACKEND_BREADTH_FIRST;
    PFFFT_Setup* off = pffft_internal_new_setup(N, tt, bf | PFFFT_BACKEND_NO_PREFETCH);
    PFFFT_Setup* on = pffft_internal_new_setup(N, tt, bf | PFFFT_BACKEND_PREFETCH);
    const int nf = cplx ? 2 * N : N;
    float *x = alloc_buf(nf), *work = alloc_buf(nf), *y = alloc_buf(nf), *z = alloc_buf(nf);
    int built = 0, k, ord, dir, inplace;
    if (!on || !off) {
        report_failure("prefetch: new_setup() returned NULL", N, cplx, 1);
        goto done;
    }
    if (pffft_internal_prefetch_passes(off) != 0) {
        report_failure("prefetch: forced off, but prefetching", N, cplx, 1);
    }
    if (pffft_internal_prefetch_passes(on) == 0) {
        goto done; /* not in this build */
    }
    built = 1;
    if (cplx) {
        /* a depth-first setup runs other passes: no flags */
        PFFFT_Setup* df =
            pffft_internal_new_setup(N, tt, PFFFT_BACKEND_NATIVE | PFFFT_BACKEND_DEPTH_FIRST | PFFFT_BACKEND_PREFETCH);
        if (df && pffft_internal_depth_first_levels(df) && pffft_internal_prefetch_passes(df) != 0) {
            report_failure("prefetch: flags on a depth-first setup", N, cplx, 1);
        }
        pffft_destroy_setup(df);
    }
    for (k = 0; k < nf; ++k) {
        x[k] = (float)urand();
    }
    for (ord = 0; ord < 2; ++ord) {
        transform_fn fn = ord ? pffft_transform_ordered : pffft_transform;
        for (dir = 0; dir < 2; ++dir) {
            const pffft_direction_t d = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
            for (inplace = 0; inplace < 2; ++inplace) {
                /* the prefetching side with scratch on the stack where that
                   is small */
                float* won = nf <= 16384 ? NULL : work;
                if (inplace) {
                    memcpy(y, x, sizeof(float) * (size_t)nf);
                    memcpy(z, x, sizeof(float) * (size_t)nf);
                    fn(on, y, y, won, d);
                    fn(off, z, z, work, d);
                } else {
                    fn(on, x, y, won, d);
                    fn(off, x, z, work, d);
                }
                if (!same_bits(y, z, nf)) {
                    report_failure("prefetching passes change the bits", N, cplx, 4 * ord + 2 * dir + inplace);
                }
            }
        }
    }
done:
    pffft_aligned_free(x);
    pffft_aligned_free(work);
    pffft_aligned_free(y);
    pffft_aligned_free(z);
    pffft_destroy_setup(on);
    pffft_destroy_setup(off);
    return built;
}

static int
test_reference(void) {
    /* the reference must agree with the defining O(N^2) sum */
    const double worst = reference_self_check(&rng_state);
    printf("reference DFT self-check: max rel err %.3g (long double epsilon %.3g)\n", worst, (double)LDBL_EPSILON);
    return worst <= 64 * LDBL_EPSILON * log2(200.0);
}

int
main(int argc, char** argv) {
    const int max_n = argc > 1 ? parse_test_size(argv[1], PFFFT_MAX_N) : DEFAULT_MAX_N;
    int cplx, N, counts[2] = {0, 0};

    if (max_n < pffft_min_fft_size(PFFFT_REAL)) {
        printf("usage: %s [max_N >= %d]\n", argv[0], pffft_min_fft_size(PFFFT_REAL));
        return 2;
    }
    printf("pffft accuracy test: simd_size %d (%s), sizes up to %d\n", pffft_simd_size(), pffft_simd_arch(), max_n);
    if (!test_reference()) {
        printf("FAIL: reference DFT disagrees with the naive DFT\n");
        return 1;
    }
    test_size_helpers();

    g_backend = PFFFT_BACKEND_NATIVE;
    for (cplx = 0; cplx < 2; ++cplx) {
        const int nmin = pffft_min_fft_size(cplx ? PFFFT_COMPLEX : PFFFT_REAL);
        for (N = nmin; N <= max_n; N += nmin) {
            if (!is_smooth(N / nmin)) {
                continue;
            }
            test_size(N, cplx);
            ++counts[cplx];
        }
    }
    printf("tested %d REAL and %d COMPLEX sizes\n", counts[0], counts[1]);
    printf("worst rel. L2 error in eps*log2(N): forward %.3f, backward %.3f, convolution %.3f (limit %.1f)\n",
           worst_fwd, worst_bwd, worst_conv, TOL_FFT);

    /* the prefetching passes, where the build has them: the same bits */
    counts[0] = counts[1] = 0;
    for (cplx = 0; cplx < 2; ++cplx) {
        const int nmin = pffft_min_fft_size(cplx ? PFFFT_COMPLEX : PFFFT_REAL);
        for (N = nmin; N <= max_n; N += nmin) {
            if (is_smooth(N / nmin)) {
                counts[cplx] += test_prefetch_size(N, cplx);
            }
        }
    }
    printf("prefetching passes: compared %d REAL and %d COMPLEX sizes\n", counts[0], counts[1]);

    /* the depth-first schedule: none where the build has it off, or
     without lanes; pffft_new_setup() uses it for larger sizes than the
     default max_N with these lanes */
    worst_fwd = worst_bwd = worst_conv = 0;
    N = test_depth_first(PFFFT_BACKEND_NATIVE, pffft_min_fft_size(PFFFT_COMPLEX), max_n);
    printf("depth-first schedule: tested %d COMPLEX sizes\n", N);
    if (N) {
        printf("worst rel. L2 error in eps*log2(N): forward %.3f, backward %.3f, convolution %.3f (limit %.1f)\n",
               worst_fwd, worst_bwd, worst_conv, TOL_FFT);
    }

    /* the 8-lane backend, where the build and the CPU have it: N must be a
     multiple of 64 (complex) or 128 (real) with N/8 = 2^a 3^b 5^c */
    {
        PFFFT_Setup* probe = pffft_internal_new_setup(128, PFFFT_REAL, PFFFT_BACKEND_W8);
        if (probe) {
            int wide_counts[2] = {0, 0};
            pffft_destroy_setup(probe);
            g_backend = PFFFT_BACKEND_W8;
            worst_fwd = worst_bwd = worst_conv = 0;
            for (cplx = 0; cplx < 2; ++cplx) {
                const int nmin = cplx ? 64 : 128;
                for (N = nmin; N <= max_n; N += nmin) {
                    if (!is_smooth(N / nmin)) {
                        continue;
                    }
                    test_size(N, cplx);
                    ++wide_counts[cplx];
                }
            }
            printf("8-lane backend: tested %d REAL and %d COMPLEX sizes\n", wide_counts[0], wide_counts[1]);
            printf("worst rel. L2 error in eps*log2(N): forward %.3f, backward %.3f, convolution %.3f (limit %.1f)\n",
                   worst_fwd, worst_bwd, worst_conv, TOL_FFT);
            {
                /* with 8 lanes and the default settings pffft_new_setup()
                 schedules these depth first from 32 KiB (4096 points) on:
                 slices that are leaves, deeper ones, and a plan that ends
                 with radix 6 (4, 4, 4, 4, 4, 6); 2048 points stay below
                 the threshold */
                static const int auto_sizes[] = {2048, 4096, 16384, 65536, 49152};
                int k;
                worst_fwd = worst_bwd = worst_conv = 0;
                N = test_depth_first(PFFFT_BACKEND_W8, 64, max_n);
                for (k = 0; k < (int)(sizeof(auto_sizes) / sizeof(auto_sizes[0])); ++k) {
                    const int t = test_depth_first_size(PFFFT_BACKEND_W8, auto_sizes[k], 0);
                    const int from_thr = 8ULL * auto_sizes[k] + 1 > t_df_min_bytes_wide;
                    N += t;
                    /* the threshold test of pffft_internal_new_setup(); the
                       plans start with radix-4 passes with
                       PFFFT_TERMINAL_RADIX8 */
                    if (t_depth_first && t_terminal_radix8 && !t && from_thr) {
                        report_failure("8 lanes: not scheduled depth first", auto_sizes[k], 1, 0);
                    }
                    if (t && !from_thr) {
                        report_failure("8 lanes: scheduled depth first below the threshold", auto_sizes[k], 1, 0);
                    }
                }
                printf("8-lane backend, depth-first schedule: tested %d COMPLEX sizes\n", N);
                if (N) {
                    printf("worst rel. L2 error in eps*log2(N): forward %.3f, backward %.3f, convolution %.3f (limit "
                           "%.1f)\n",
                           worst_fwd, worst_bwd, worst_conv, TOL_FFT);
                }
            }
        } else {
            printf("8-lane backend: not available in this build or on this CPU\n");
        }
        g_backend = PFFFT_BACKEND_AUTO;
    }
    release_twiddles();

    if (num_failures) {
        printf("%d FAILURES\n", num_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
