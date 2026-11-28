// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  The exact transform that pffft's float results are measured against, shared
  by tests/test_pffft_accuracy.c and benchmarks/fft_accuracy.c:

    forward:  X[k] = sum_n x[n] exp(-2*pi*i*n*k/N)
    backward: x[n] = sum_k X[k] exp(+2*pi*i*n*k/N)   (unnormalized)

  computed in long double by a recursive mixed-radix decimation in time with
  generic radix butterflies, from twiddles whose angle is reduced exactly in
  integer arithmetic. Long double has 64 bits of mantissa on x86 (GCC,
  Clang), 113 on aarch64 Linux and 53 on MSVC (where it is double): at least
  29 bits more than float everywhere, so the error of the reference is far
  below pffft's and does not correlate with it.

  Also the conversions between pffft's ordered layouts and full spectra, and
  the xorshift64 generator of the inputs. Every function is static: each
  includer has its own copy and its own twiddle table.
*/

#ifndef PFFFT_FFT_REFERENCE_H
#define PFFFT_FFT_REFERENCE_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long double ldouble;

typedef struct {
    ldouble re, im;
} cld;

static cld*
alloc_reference(int n) {
    if (n <= 0) {
        (void)fputs("Invalid reference buffer size\n", stderr);
        exit(EXIT_FAILURE);
    }
    cld* buffer = calloc((size_t)n, sizeof(*buffer));
    if (!buffer) {
        (void)fputs("Could not allocate reference buffer\n", stderr);
        exit(EXIT_FAILURE);
    }
    return buffer;
}

/* xorshift64 -- reproducible on every platform; state must not be zero */
static double
xorshift_urand(unsigned long long* state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (double)(*state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
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
   integer arithmetic so that its magnitude never exceeds pi/4. The table of
   the last (N, sign) is kept: transform all inputs of one size and direction
   before the next. */
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

/* frees the twiddle table */
static void
release_twiddles(void) {
    free(tw_table);
    tw_table = NULL;
    tw_n = tw_sign = 0;
}

/* recursive mixed-radix decimation in time, generic radix butterflies */
// Each call divides n by a factor of at least two.
static void
// NOLINTNEXTLINE(misc-no-recursion)
ref_rec(int n, int N, const cld* in, int offset, int istride, cld* out, cld* tmp) {
    int p, m, r, k, q;
    if (n == 1) {
        if (offset < 0 || offset >= N) {
            (void)fputs("Invalid reference DFT offset\n", stderr);
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

/* the DFT of N values; sign -1 forward, +1 backward */
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

/* The largest relative L2 distance between ref_dft() and the defining
   O(N^2) sum over all N up to 200, on inputs from *state; about
   LDBL_EPSILON when the reference is right. */
static double
reference_self_check(unsigned long long* state) {
    int n, k;
    double worst = 0;
    for (n = 1; n <= 200; ++n) {
        cld *in = alloc_reference(n), *o1 = alloc_reference(n), *o2 = alloc_reference(n);
        ldouble num = 0, den = 0;
        for (k = 0; k < n; ++k) {
            in[k].re = xorshift_urand(state);
            in[k].im = xorshift_urand(state);
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
    return worst;
}

/* ---------------- layouts and errors ---------------- */

/* adds the squared L2 distance of y to ref to *num and the squared norm of
   ref to *den; y holds n complex values interleaved (cplx) or n reals */
static void
add_err_sums(int n, const float* y, const cld* ref, int cplx, ldouble* num, ldouble* den) {
    int k;
    for (k = 0; k < n; ++k) {
        ldouble dr = (cplx ? (ldouble)y[(ptrdiff_t)2 * k] : (ldouble)y[k]) - ref[k].re;
        ldouble di = (cplx ? (ldouble)y[2 * k + 1] : 0) - (cplx ? ref[k].im : 0);
        *num += dr * dr + di * di;
        *den += ref[k].re * ref[k].re + (cplx ? ref[k].im * ref[k].im : 0);
    }
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

#endif /* PFFFT_FFT_REFERENCE_H */
