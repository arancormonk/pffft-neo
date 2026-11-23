// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Accuracy and contract test for pffft.

  Every valid transform size up to a limit is checked against an
  independent reference DFT computed in long double:

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

typedef long double ldouble;

typedef struct {
    ldouble re, im;
} cld;

static cld*
alloc_reference(int n) {
    if (n <= 0 || (size_t)n > SIZE_MAX / sizeof(cld)) {
        fputs("Invalid reference buffer size\n", stderr);
        exit(EXIT_FAILURE);
    }
    cld* buffer = malloc(sizeof(*buffer) * (size_t)n);
    if (!buffer) {
        fputs("Could not allocate reference buffer\n", stderr);
        exit(EXIT_FAILURE);
    }
    return buffer;
}

static int num_failures = 0;

static void
report_failure(const char* what, int N, int cplx, double err) {
    ++num_failures;
    if (num_failures <= MAX_REPORTED_FAILURES) {
        printf("FAIL: %s %s N=%d rel_err=%g (%.2f eps*log2N)\n", what, cplx ? "COMPLEX" : "REAL", N, err,
               err / (FLT_EPSILON * (N > 1 ? log2((double)N) : 1.0)));
    }
}

/* xorshift64 -- reproducible on every platform */
static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;

static double
urand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static cld
cmul(cld a, cld b) {
    cld r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* ---------------- reference DFT ---------------- */

static cld* tw_table = NULL;
static int tw_n = 0, tw_sign = 0;

/* tw_table[t] = exp(sign*2*pi*i*t/N), with the angle reduced exactly in
   integer arithmetic so that its magnitude never exceeds pi/4 */
static void
make_twiddles(int N, int sign) {
    const ldouble two_pi = 6.283185307179586476925286766559005768L;
    int t;
    if (tw_n == N && tw_sign == sign) {
        return;
    }
    free(tw_table);
    tw_table = alloc_reference(N);
    tw_n = N;
    tw_sign = sign;
    for (t = 0; t < N; ++t) {
        /* angle = 2*pi*t/N = q*pi/2 + 2*pi*r/(4N), with |r| <= N/2 */
        long long t4 = 4LL * t;
        long long q = (t4 + N / 2) / N; /* nearest quarter turn */
        long long r = t4 - q * N;       /* remainder in [-N/2, N/2] */
        ldouble a = two_pi * (ldouble)r / (ldouble)(4LL * N);
        ldouble c = cosl(a), s = sinl(a), cr, sr;
        switch (q & 3) {
            case 0:
                cr = c;
                sr = s;
                break;
            case 1:
                cr = -s;
                sr = c;
                break;
            case 2:
                cr = -c;
                sr = -s;
                break;
            default:
                cr = s;
                sr = -c;
                break;
        }
        tw_table[t].re = cr;
        tw_table[t].im = sign * sr;
    }
}

/* recursive mixed-radix decimation in time, generic radix butterflies */
// Each call divides n by a factor of at least two.
static void
// NOLINTNEXTLINE(misc-no-recursion)
ref_rec(int n, int N, const cld* in, int offset, int istride, cld* out, cld* tmp) {
    int p, m, r, k, q;
    if (n == 1) {
        if (offset < 0 || offset >= N) {
            fputs("Invalid reference DFT offset\n", stderr);
            exit(EXIT_FAILURE);
        }
        out[0] = in[offset];
        return;
    }
    for (p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            break;
        }
    }
    if (n % p) {
        p = n;
    }
    m = n / p;
    for (r = 0; r < p; ++r) {
        ref_rec(m, N, in, offset + r * istride, istride * p, out + (ptrdiff_t)r * m, tmp);
    }
    for (k = 0; k < m; ++k) {
        for (q = 0; q < p; ++q) {
            cld s = {0, 0};
            for (r = 0; r < p; ++r) {
                int t = (int)(((long long)r * (k + m * q)) % n) * (N / n);
                cld v = cmul(tw_table[t], out[r * m + k]);
                s.re += v.re;
                s.im += v.im;
            }
            tmp[k + m * q] = s;
        }
    }
    memcpy(out, tmp, sizeof(cld) * n);
}

static void
ref_dft(int N, const cld* in, cld* out, int sign) {
    cld* tmp = alloc_reference(N);
    make_twiddles(N, sign);
    ref_rec(N, N, in, 0, 1, out, tmp);
    free(tmp);
}

static void
naive_dft(int N, const cld* in, cld* out, int sign) {
    int k, n;
    make_twiddles(N, sign);
    for (k = 0; k < N; ++k) {
        cld s = {0, 0};
        for (n = 0; n < N; ++n) {
            cld v = cmul(in[n], tw_table[(int)(((long long)n * k) % N)]);
            s.re += v.re;
            s.im += v.im;
        }
        out[k] = s;
    }
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
    int k;
    for (k = 0; k < n; ++k) {
        ldouble dr = (cplx ? (ldouble)y[(ptrdiff_t)2 * k] : (ldouble)y[k]) - ref[k].re;
        ldouble di = (cplx ? (ldouble)y[2 * k + 1] : 0) - (cplx ? ref[k].im : 0);
        num += dr * dr + di * di;
        den += ref[k].re * ref[k].re + (cplx ? ref[k].im * ref[k].im : 0);
    }
    return (double)(den > 0 ? sqrtl(num / den) : sqrtl(num));
}

static void
to_cld(int N, const float* x, int cplx, cld* out) {
    int k;
    for (k = 0; k < N; ++k) {
        out[k].re = cplx ? (ldouble)x[(ptrdiff_t)2 * k] : (ldouble)x[k];
        out[k].im = cplx ? (ldouble)x[2 * k + 1] : 0;
    }
}

/* ordered REAL layout (as complex pairs): slot 0 = F(0) + i*F(N/2) */
static void
real_spectrum_to_ordered(int N, const cld* F, cld* out) {
    int k;
    out[0].re = F[0].re;
    out[0].im = F[N / 2].re;
    for (k = 1; k < N / 2; ++k) {
        out[k] = F[k];
    }
}

/* ordered REAL layout -> full Hermitian spectrum */
static void
ordered_to_real_spectrum(int N, const float* in, cld* F) {
    int k;
    F[0].re = in[0];
    F[0].im = 0;
    F[N / 2].re = in[1];
    F[N / 2].im = 0;
    for (k = 1; k < N / 2; ++k) {
        F[k].re = in[(ptrdiff_t)2 * k];
        F[k].im = in[2 * k + 1];
        F[N - k].re = in[(ptrdiff_t)2 * k];
        F[N - k].im = -(ldouble)in[2 * k + 1];
    }
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

/* every complex size up to max_n whose plan allows the depth-first
   schedule, forced; returns their number */
static int
test_depth_first(int backend, int nmin, int max_n) {
    int N, tested = 0;
    for (N = nmin; N <= max_n; N += nmin) {
        if (is_smooth(N / nmin)) {
            tested += test_depth_first_size(backend, N, 1);
        }
    }
    return tested;
}

static int
test_reference(void) {
    /* the reference must agree with the defining O(N^2) sum */
    int n, k;
    double worst = 0;
    for (n = 1; n <= 200; ++n) {
        cld *in = alloc_reference(n), *o1 = alloc_reference(n), *o2 = alloc_reference(n);
        ldouble num = 0, den = 0;
        for (k = 0; k < n; ++k) {
            in[k].re = urand();
            in[k].im = urand();
        }
        ref_dft(n, in, o1, -1);
        naive_dft(n, in, o2, -1);
        for (k = 0; k < n; ++k) {
            ldouble dr = o1[k].re - o2[k].re, di = o1[k].im - o2[k].im;
            num += dr * dr + di * di;
            den += o2[k].re * o2[k].re + o2[k].im * o2[k].im;
        }
        if (sqrtl(num / den) > worst) {
            worst = (double)sqrtl(num / den);
        }
        free(in);
        free(o1);
        free(o2);
    }
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
                /* with 8 lanes pffft_new_setup() schedules these two depth
                 first, one with slices that are leaves and one deeper */
                static const int auto_sizes[] = {16384, 65536};
                worst_fwd = worst_bwd = worst_conv = 0;
                N = test_depth_first(PFFFT_BACKEND_W8, 64, max_n);
                N += test_depth_first_size(PFFFT_BACKEND_W8, auto_sizes[0], 0);
                N += test_depth_first_size(PFFFT_BACKEND_W8, auto_sizes[1], 0);
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
    free(tw_table);

    if (num_failures) {
        printf("%d FAILURES\n", num_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
