// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  The per-backend part of test_pffft_passes.c, which includes this file once
  for each backend whose codelets it can call. Deliberately no include
  guard. Before each inclusion the including file defines:

    T_LANES      floats per vector (SIMD_SZ, or 8 for the 8-lane backend)
    T_V          the vector type (v4sf, or pffft_v8sf)
    T_FN(name)   the static library function `name` of this backend
                 (`name`, or `pffft_w8_##name`)
    T_FUSED      1 when the backend has the fused layout passes
                 (PF_FUSED_LAYOUT: passf*_il_ps, passf4_ol_ps,
                 cfftf1_il_ps, cfftf1_tail_ol_ps), else 0
    T_NAME(name) the name of a test function of this backend
                 (`name##_native`, or `name##_w8`)

  and T_EPS, T_CHECK(), t_check() and the helpers t_alloc(), t_free(),
  t_fill(), t_max_err(), t_within(), t_is_235() and (with
  PFFFT_TERMINAL_RADIX8) t_is_435() are available. Buffers are indexed as
  float arrays, T_LANES floats per vector; the vector macros (VLD, VST,
  VADD, ...) are not used here, since after the 8-lane inclusion they are
  the 256-bit ones. This file undefines the T_* backend macros at its end.

  T_NAME(run_backend_tests) runs every test of this file for the backend;
  each test adds the number of its failed checks to the return value.
*/

/* E(n): FFTPACK's factor list for the per-lane length n (the try-lists of
   rffti1_ps and cffti1_ps), then the plan rule restated here. The factors
   go to f[0..]; returns their number. */
static int
T_NAME(expected_plan)(int n, int cplx, int* f) {
    static const int cplx_try[] = {5, 3, 4, 2, 0};
    static const int real_try[] = {4, 2, 3, 5, 0};
    int ifac[IFAC_MAX_SIZE];
    int nf = T_FN(decompose)(n, ifac, cplx ? cplx_try : real_try), k;
    for (k = 0; k < nf; ++k) {
        f[k] = ifac[2 + k];
    }
#if PFFFT_TERMINAL_RADIX8
    /* a 2 first and a 4: drop both, the rest in order, then an 8 */
    if (nf >= 2 && f[0] == 2) {
        int j = 1, m = 0;
        while (j < nf && f[j] != 4) {
            ++j;
        }
        if (j < nf) {
            for (k = 1; k < nf; ++k) {
                if (k != j) {
                    f[m++] = f[k];
                }
            }
            f[m] = 8;
            nf = m + 1;
        }
    }
#else
    /* the single radix-8 pass of complex n == 8 */
    if (cplx && n == 8) {
        f[0] = 8;
        nf = 1;
    }
#endif
    return nf;
}

/* the checks of test_plan_rule on the factor list ifac of n; nf is the
   number of factors the function under test returned (or its ifac[1]) */
static int
T_NAME(check_plan)(int n, int cplx, const int* ifac, int nf) {
    int f[IFAC_MAX_SIZE];
    const int ne = T_NAME(expected_plan)(n, cplx, f);
    int fails = 0, k, prod = 1, same = 1;
    fails += T_CHECK(ifac[0] == n);
    fails += T_CHECK(ifac[1] == ne);
    fails += T_CHECK(nf == ne);
    for (k = 0; k < ne && k < ifac[1]; ++k) {
        same = same && ifac[2 + k] == f[k];
        prod *= ifac[2 + k];
    }
    fails += T_CHECK(same);
    fails += T_CHECK(prod == n);
#if PFFFT_TERMINAL_RADIX8
    {
        /* independent of E: the 8 is the pass with one element per leg, and
       no stand-alone radix-2 pass is left where a radix-4 pass could
       absorb it */
        int has2 = 0, has4 = 0, has8 = 0;
        for (k = 0; k < ifac[1]; ++k) {
            has2 |= ifac[2 + k] == 2;
            has4 |= ifac[2 + k] == 4;
            has8 |= ifac[2 + k] == 8;
            fails += T_CHECK(ifac[2 + k] != 8 || k == ifac[1] - 1);
        }
        fails += T_CHECK(!(has2 && has8));
        fails += T_CHECK(!(has2 && has4));
    }
#endif
    if (fails) {
        printf("  (%s n = %d)\n", cplx ? "complex" : "real", n);
    }
    return fails;
}

/* the plan of every per-lane length: cffti1_ps and rffti1_ps against E(n) */
static int
T_NAME(test_plan_rule)(void) {
    const int nmax = 8192;
    float* wa = t_alloc(2 * nmax + 16);
    int ifac[IFAC_MAX_SIZE];
    int fails = 0, cplx, n;
    if (!wa) {
        return T_CHECK(wa != NULL);
    }
    for (cplx = 1; cplx >= 0; --cplx) {
        for (n = 2; n <= nmax; ++n) {
            if (!t_is_235(n)) {
                continue;
            }
            if (cplx) {
                T_FN(cffti1_ps)(n, wa, ifac);
            } else {
                T_FN(rffti1_ps)(n, wa, ifac);
            }
            fails += T_NAME(check_plan)(n, cplx, ifac, ifac[1]);
        }
    }
#if PFFFT_TERMINAL_RADIX8
    /* longer lists, the rule alone (no twiddles computed) */
    for (cplx = 1; cplx >= 0; --cplx) {
        static const int cplx_try[] = {5, 3, 4, 2, 0};
        static const int real_try[] = {4, 2, 3, 5, 0};
        int k, m;
        for (k = 1; k <= 24; ++k) {
            for (m = 1; m <= 5; m += 2) { /* 2^k, 3*2^k, 5*2^k above nmax */
                int nf;
                n = m << k;
                if (n <= nmax || n > (1 << 24)) {
                    continue;
                }
                T_FN(decompose)(n, ifac, cplx ? cplx_try : real_try);
                nf = T_FN(terminal_radix8_plan)(ifac);
                fails += T_NAME(check_plan)(n, cplx, ifac, nf);
            }
        }
    }
#endif
    t_free(wa);
    return fails;
}

#if PFFFT_TERMINAL_RADIX8
/* complex radix 8 at one element per leg: passf8_ps against the radix-2
   pass and the radix-4 pass it replaces, block by block */
static int
T_NAME(test_radix8_complex)(void) {
    const int lmax = 512, V = T_LANES;
    float *cc = t_alloc((size_t)16 * lmax * V), *ch = t_alloc((size_t)16 * lmax * V);
    float *x = t_alloc((size_t)16 * V), *t = t_alloc((size_t)16 * V), *y = t_alloc((size_t)16 * V), wa2[8];
    int fails = 0, s, i;
    if (!cc || !ch || !x || !t || !y) {
        fails += T_CHECK(cc && ch && x && t && y);
        goto done;
    }
    for (i = 0; i < 4; ++i) {
        T_FN(twiddle_cos_sin)(i, 8, &wa2[(ptrdiff_t)2 * i], &wa2[2 * i + 1]);
    }
    for (s = 0; s < 2; ++s) {
        const float fsign = s ? 1.f : -1.f;
        int l1;
        for (l1 = 1; l1 <= lmax; ++l1) {
            int k, bad = 0;
            if (!t_is_435(l1)) {
                continue;
            }
            t_fill(cc, (size_t)16 * l1 * V);
            T_FN(passf8_ps)(2, l1, (const T_V*)cc, (T_V*)ch, fsign);
            for (k = 0; k < l1; ++k) {
                int m, r, l;
                for (i = 0; i < 16 * V; ++i) {
                    x[i] = cc[16 * k * V + i];
                }
                T_FN(passf2_ps)(8, 1, (const T_V*)x, (T_V*)t, wa2, fsign);
                T_FN(passf4_ps)(2, 2, (const T_V*)t, (T_V*)y, wa2, wa2, wa2, fsign);
                for (m = 0; m < 8; ++m) {
                    for (r = 0; r < 2; ++r) {
                        for (l = 0; l < V; ++l) {
                            bad += !t_within(fabsf(ch[(2 * (k + m * l1) + r) * V + l] - y[(2 * m + r) * V + l]),
                                             128 * T_EPS);
                        }
                    }
                }
            }
            if (bad) {
                printf("  (passf8_ps: fsign %+g, l1 %d: %d values off)\n", fsign, l1, bad);
            }
            fails += T_CHECK(bad == 0);
        }
    }
done:
    t_free(cc);
    t_free(ch);
    t_free(x);
    t_free(t);
    t_free(y);
    return fails;
}

/* real radix 8 at ido == 1: radf8_ps and radb8_ps against the radix-4 and
   radix-2 passes they replace, block by block */
static int
T_NAME(test_radix8_real)(void) {
    static const int l1s[] = {1, 2, 3, 4, 5, 16, 64, 256};
    const int lmax = 256, V = T_LANES;
    float *cc = t_alloc((size_t)8 * lmax * V), *ch = t_alloc((size_t)8 * lmax * V);
    float *x = t_alloc((size_t)8 * V), *t = t_alloc((size_t)8 * V), *y = t_alloc((size_t)8 * V), wa[7] = {0};
    int fails = 0, il;
    if (!cc || !ch || !x || !t || !y) {
        fails += T_CHECK(cc && ch && x && t && y);
        goto done;
    }
    T_FN(twiddle_cos_sin)(1, 8, &wa[0], &wa[1]);
    for (il = 0; il < (int)(sizeof(l1s) / sizeof(l1s[0])); ++il) {
        const int l1 = l1s[il];
        int k, j, m, l, i, bad;
        /* forward */
        t_fill(cc, (size_t)8 * l1 * V);
        T_FN(radf8_ps)(1, l1, (const T_V*)cc, (T_V*)ch);
        bad = 0;
        for (k = 0; k < l1; ++k) {
            for (j = 0; j < 8; ++j) {
                for (l = 0; l < V; ++l) {
                    x[j * V + l] = cc[(k + j * l1) * V + l];
                }
            }
            T_FN(radf4_ps)(1, 2, (const T_V*)x, (T_V*)t, wa + 4, wa + 5, wa + 6);
            T_FN(radf2_ps)(4, 1, (const T_V*)t, (T_V*)y, wa);
            for (m = 0; m < 8; ++m) {
                for (l = 0; l < V; ++l) {
                    bad += !t_within(fabsf(ch[(8 * k + m) * V + l] - y[m * V + l]), 128 * T_EPS);
                }
            }
        }
        if (bad) {
            printf("  (radf8_ps: l1 %d: %d values off)\n", l1, bad);
        }
        fails += T_CHECK(bad == 0);
        /* backward */
        t_fill(cc, (size_t)8 * l1 * V);
        T_FN(radb8_ps)(1, l1, (const T_V*)cc, (T_V*)ch);
        bad = 0;
        for (k = 0; k < l1; ++k) {
            for (i = 0; i < 8 * V; ++i) {
                x[i] = cc[8 * k * V + i];
            }
            T_FN(radb2_ps)(4, 1, (const T_V*)x, (T_V*)t, wa);
            T_FN(radb4_ps)(1, 2, (const T_V*)t, (T_V*)y, wa + 4, wa + 5, wa + 6);
            for (j = 0; j < 8; ++j) {
                for (l = 0; l < V; ++l) {
                    bad += !t_within(fabsf(ch[(k + j * l1) * V + l] - y[j * V + l]), 256 * T_EPS);
                }
            }
        }
        if (bad) {
            printf("  (radb8_ps: l1 %d: %d values off)\n", l1, bad);
        }
        fails += T_CHECK(bad == 0);
    }
done:
    t_free(cc);
    t_free(ch);
    t_free(x);
    t_free(t);
    t_free(y);
    return fails;
}
#endif /* PFFFT_TERMINAL_RADIX8 */

/* the half-complex real DFT (FFTPACK's layout) of x[0], x[stride], ...,
   L values, in double */
static void
T_NAME(hc_dft)(const double* x, int L, int stride, double* hc) {
    int f, t;
    for (f = 0; 2 * f <= L; ++f) {
        double re = 0, im = 0;
        for (t = 0; t < L; ++t) {
            const double a = 2 * M_PI * (double)((long)f * t % L) / L;
            re += x[t * stride] * cos(a);
            im -= x[t * stride] * sin(a);
        }
        if (f == 0) {
            hc[0] = re;
        } else if (2 * f == L) {
            hc[L - 1] = re;
        } else {
            hc[2 * f - 1] = re;
            hc[(ptrdiff_t)2 * f] = im;
        }
    }
}

/* The real radix-3 and radix-5 passes, one pass at a time, against a DFT
   in double, for odd and (with PFFFT_TERMINAL_RADIX8, the only plans that
   give them one) even ido: an even ido adds the Nyquist elements, and
   ido == 2 has only those and the DC elements. Group k of a pass of
   radix p holds the spectra Y_j, of length ido, of z[j], z[j + p], ... for
   a real z of length L = p*ido: forward they go to the spectrum X of z,
   and backward X goes to p Y_j (unscaled, as all passes). */
static int
T_NAME(test_real_odd_radix)(void) {
#if PFFFT_TERMINAL_RADIX8
    static const int idos[] = {1, 2, 3, 4, 5, 6, 8, 10, 16};
#else
    static const int idos[] = {1, 3, 5, 7, 9, 15};
#endif
    static const int l1s[] = {1, 3};

    enum { IMAX = 16, PMAX = 5, LMAX = 3 };

    const int V = T_LANES;
    const size_t nf = (size_t)PMAX * IMAX * LMAX * V; /* floats per buffer */
    float *cc = t_alloc(nf), *ch = t_alloc(nf), *wa = t_alloc((size_t)(PMAX - 1) * IMAX),
          *r = t_alloc((size_t)PMAX * IMAX);
    double *want = (double*)malloc(nf * sizeof(double)), z[PMAX * IMAX], X[PMAX * IMAX], Y[PMAX * IMAX];
    const float* w[PMAX - 1];
    int fails = 0, p, ii, il, dir;
    if (!cc || !ch || !wa || !r || !want) {
        fails += T_CHECK(cc && ch && wa && r && want);
        goto done;
    }
    for (p = 0; p < PMAX - 1; ++p) {
        w[p] = wa + (ptrdiff_t)p * IMAX;
    }
    for (p = 3; p <= 5; p += 2) {
        for (ii = 0; ii < (int)(sizeof(idos) / sizeof(idos[0])); ++ii) {
            const int ido = idos[ii], L = p * ido;
            int j, f;
            /* the twiddles of leg j as rffti1_ps stores them: cos and sin
               of 2 pi j f/L, f = 1 .. (ido-1)/2 (the forward pass takes
               their conjugates) */
            for (j = 1; j < p; ++j) {
                for (f = 1; 2 * f < ido; ++f) {
                    T_FN(twiddle_cos_sin)(j * f, L, &wa[(ptrdiff_t)(j - 1) * IMAX + (ptrdiff_t)2 * f - 2],
                                          &wa[(ptrdiff_t)(j - 1) * IMAX + (ptrdiff_t)2 * f - 1]);
                }
            }
            for (il = 0; il < (int)(sizeof(l1s) / sizeof(l1s[0])); ++il) {
                const int l1 = l1s[il];
                for (dir = 0; dir < 2; ++dir) {
                    int k, l, i, bad = 0;
                    double worst = 0;
                    for (k = 0; k < l1; ++k) {
                        for (l = 0; l < V; ++l) {
                            t_fill(r, (size_t)L);
                            for (i = 0; i < L; ++i) {
                                z[i] = r[i];
                            }
                            T_NAME(hc_dft)(z, L, 1, X);
                            for (j = 0; j < p; ++j) {
                                T_NAME(hc_dft)(z + j, ido, p, Y + (ptrdiff_t)j * ido);
                            }
                            for (j = 0; j < p; ++j) {
                                for (i = 0; i < ido; ++i) {
                                    const size_t leg = ((size_t)(k + j * l1) * ido + i) * V + l;
                                    const size_t blk = ((size_t)k * L + (size_t)j * ido + i) * V + l;
                                    if (dir == 0) {
                                        cc[leg] = (float)Y[j * ido + i];
                                        want[blk] = X[j * ido + i];
                                    } else {
                                        cc[blk] = (float)X[j * ido + i];
                                        want[leg] = p * Y[j * ido + i];
                                    }
                                }
                            }
                        }
                    }
                    if (p == 3) {
                        if (dir == 0) {
                            T_FN(radf3_ps)(ido, l1, (const T_V*)cc, (T_V*)ch, w[0], w[1]);
                        } else {
                            T_FN(radb3_ps)(ido, l1, (const T_V*)cc, (T_V*)ch, w[0], w[1]);
                        }
                    } else if (dir == 0) {
                        T_FN(radf5_ps)(ido, l1, (const T_V*)cc, (T_V*)ch, w[0], w[1], w[2], w[3]);
                    } else {
                        T_FN(radb5_ps)(ido, l1, (const T_V*)cc, (T_V*)ch, w[0], w[1], w[2], w[3]);
                    }
                    /* |X| and |p Y| are at most L */
                    for (i = 0; i < l1 * L * V; ++i) {
                        const double e = fabs(ch[i] - want[i]) / L;
                        worst = t_max_err(worst, e);
                        bad += !t_within(e, 8 * T_EPS);
                    }
                    if (bad) {
                        printf("  (rad%c%d_ps: ido %d, l1 %d: %d values off, worst %.3g eps)\n", dir ? 'b' : 'f', p,
                               ido, l1, bad, worst / T_EPS);
                    }
                    fails += T_CHECK(bad == 0);
                }
            }
        }
    }
done:
    t_free(cc);
    t_free(ch);
    t_free(wa);
    t_free(r);
    free(want);
    return fails;
}

#if PFFFT_TERMINAL_RADIX8 && T_FUSED
/* passf8_ol_ps: the bits of passf8_ps, interleaved as PF_INTERLEAVE2 does */
static int
T_NAME(test_radix8_complex_ol)(void) {
    static const int l1s[] = {1, 2, 3, 4, 5, 16, 60};
    const int lmax = 60, V = T_LANES;
    float *cc = t_alloc((size_t)16 * lmax * V), *ch = t_alloc((size_t)16 * lmax * V),
          *out = t_alloc((size_t)16 * lmax * V);
    int fails = 0, s, il;
    if (!cc || !ch || !out) {
        fails += T_CHECK(cc && ch && out);
        goto done;
    }
    for (s = 0; s < 2; ++s) {
        const float fsign = s ? 1.f : -1.f;
        for (il = 0; il < (int)(sizeof(l1s) / sizeof(l1s[0])); ++il) {
            const int l1 = l1s[il];
            int p, bad = 0;
            t_fill(cc, (size_t)16 * l1 * V);
            T_FN(passf8_ps)(2, l1, (const T_V*)cc, (T_V*)ch, fsign);
            T_FN(passf8_ol_ps)(l1, (const T_V*)cc, (T_V*)out, fsign);
            for (p = 0; p < 8 * l1; ++p) {
                const float *a = ch + (ptrdiff_t)2 * p * V, *b = a + V;
                float want[16];
                /* 4 lanes: a0 b0 a1 b1 | a2 b2 a3 b3;
           8 lanes: a0 b0 a1 b1 a4 b4 a5 b5 | a2 b2 a3 b3 a6 b6 a7 b7 */
                if (V == 4) {
                    want[0] = a[0];
                    want[1] = b[0];
                    want[2] = a[1];
                    want[3] = b[1];
                    want[4] = a[2];
                    want[5] = b[2];
                    want[6] = a[3];
                    want[7] = b[3];
                } else {
                    want[0] = a[0];
                    want[1] = b[0];
                    want[2] = a[1];
                    want[3] = b[1];
                    want[4] = a[4];
                    want[5] = b[4];
                    want[6] = a[5];
                    want[7] = b[5];
                    want[8] = a[2];
                    want[9] = b[2];
                    want[10] = a[3];
                    want[11] = b[3];
                    want[12] = a[6];
                    want[13] = b[6];
                    want[14] = a[7];
                    want[15] = b[7];
                }
                // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison,cert-exp42-c,cert-flp37-c) -- bit-exact layout check.
                bad += memcmp(out + (ptrdiff_t)2 * p * V, want, (size_t)2 * V * sizeof(float)) != 0;
            }
            if (bad) {
                printf("  (passf8_ol_ps: fsign %+g, l1 %d: %d pairs differ)\n", fsign, l1, bad);
            }
            fails += T_CHECK(bad == 0);
        }
    }
done:
    t_free(cc);
    t_free(ch);
    t_free(out);
    return fails;
}
#endif

#if PFFFT_INPLACE_FIRST_PASS && T_FUSED
/* the first pass (l1 == 1) of radix R, idot = 2 * (elements per leg): the
   regular codelet from in to out, then its in-place wrapper on buf, a copy
   of in; il selects the deinterleaving codelets (passfR_il_ps) */
static void
T_NAME(run_inplace_pair)(int R, int il, int idot, const float* in, float* out, float* buf, float* const* w,
                         float fsign) {
    const T_V* x = (const T_V*)in;
    T_V *y = (T_V*)out, *b = (T_V*)buf;
    memcpy(buf, in, (size_t)R * idot * T_LANES * sizeof(float));
    if (!il) {
        switch (R) {
            case 2:
                T_FN(passf2_ps)(idot, 1, x, y, w[0], fsign);
                T_FN(passf2_ip_ps)(idot, b, w[0], fsign);
                break;
            case 3:
                T_FN(passf3_ps)(idot, 1, x, y, w[0], w[1], fsign);
                T_FN(passf3_ip_ps)(idot, b, w[0], w[1], fsign);
                break;
            case 4:
                T_FN(passf4_ps)(idot, 1, x, y, w[0], w[1], w[2], fsign);
                T_FN(passf4_ip_ps)(idot, b, w[0], w[1], w[2], fsign);
                break;
            default:
                T_FN(passf5_ps)(idot, 1, x, y, w[0], w[1], w[2], w[3], fsign);
                T_FN(passf5_ip_ps)(idot, b, w[0], w[1], w[2], w[3], fsign);
                break;
        }
    } else {
        switch (R) {
            case 2:
                T_FN(passf2_il_ps)(idot, x, y, w[0], fsign);
                T_FN(passf2_il_ip_ps)(idot, b, w[0], fsign);
                break;
            case 3:
                T_FN(passf3_il_ps)(idot, x, y, w[0], w[1], fsign);
                T_FN(passf3_il_ip_ps)(idot, b, w[0], w[1], fsign);
                break;
            case 4:
                T_FN(passf4_il_ps)(idot, x, y, w[0], w[1], w[2], fsign);
                T_FN(passf4_il_ip_ps)(idot, b, w[0], w[1], w[2], fsign);
                break;
            default:
                T_FN(passf5_il_ps)(idot, x, y, w[0], w[1], w[2], w[3], fsign);
                T_FN(passf5_il_ip_ps)(idot, b, w[0], w[1], w[2], w[3], fsign);
                break;
        }
    }
}

/* the in-place first-pass wrappers have the bits of the regular codelets.
   The plain ones start at two elements per leg: the plain radix-2 codelet
   with one element per leg is not safe in place and its wrapper asserts
   that it is not reached */
static int
T_NAME(test_inplace_codelets)(void) {
    static const int ms_plain[] = {2, 3, 4, 5, 8, 15, 16, 64};
    static const int ms_il[] = {1, 2, 3, 4, 5, 8, 15, 16, 64};
    const int mmax = 64, V = T_LANES;
    float *in = t_alloc((size_t)10 * mmax * V), *out = t_alloc((size_t)10 * mmax * V),
          *buf = t_alloc((size_t)10 * mmax * V);
    float* w[4] = {NULL, NULL, NULL, NULL};
    int fails = 0, R, s, il, j;
    for (j = 0; j < 4; ++j) {
        w[j] = t_alloc((size_t)2 * mmax);
    }
    if (!in || !out || !buf || !w[0] || !w[1] || !w[2] || !w[3]) {
        fails += T_CHECK(in && out && buf && w[0] && w[1] && w[2] && w[3]);
        goto done;
    }
    for (j = 0; j < 4; ++j) {
        t_fill(w[j], (size_t)2 * mmax);
    }
    for (R = 2; R <= 5; ++R) {
        for (s = 0; s < 2; ++s) {
            const float fsign = s ? 1.f : -1.f;
            for (il = 0; il < 2; ++il) {
                const int* ms = il ? ms_il : ms_plain;
                const int nm =
                    il ? (int)(sizeof(ms_il) / sizeof(ms_il[0])) : (int)(sizeof(ms_plain) / sizeof(ms_plain[0]));
                int im;
                for (im = 0; im < nm; ++im) {
                    const int m = ms[im], idot = 2 * m;
                    int same;
                    t_fill(in, (size_t)2 * R * m * V);
                    T_NAME(run_inplace_pair)(R, il, idot, in, out, buf, w, fsign);
                    same = memcmp(buf, out, (size_t)2 * R * m * V * sizeof(float)) == 0;
                    if (!same) {
                        printf("  (passf%d%s_ip_ps: fsign %+g, %d elements per leg: differs from passf%d%s_ps)\n", R,
                               il ? "_il" : "", fsign, m, R, il ? "_il" : "");
                    }
                    fails += T_CHECK(same);
                }
            }
        }
    }
done:
    t_free(in);
    t_free(out);
    t_free(buf);
    for (j = 0; j < 4; ++j) {
        t_free(w[j]);
    }
    return fails;
}
#endif

/* negates the imaginary parts of the nvec vectors of x: the vectors at odd
   positions (the passes' own layout, real and imaginary part in a vector
   each), or with il the floats at odd positions (interleaved complex) */
static void
T_NAME(conjugate)(float* x, size_t nvec, int il) {
    const size_t V = T_LANES;
    size_t i, l;
    for (i = 0; i < nvec; ++i) {
        for (l = 0; l < V; ++l) {
            if (il ? (l & 1) : (i & 1)) {
                x[i * V + l] = -x[i * V + l];
            }
        }
    }
}

/* the kinds of complex pass functions for test_pass_directions */
enum {
    T_NAME(PASS_PLAIN), /* passfR_ps */
    T_NAME(PASS_IL),    /* passfR_il_ps: interleaved input */
    T_NAME(PASS_IP),    /* passfR_ip_ps: in place */
    T_NAME(PASS_IL_IP), /* passfR_il_ip_ps: interleaved input, in place */
    T_NAME(PASS_OL),    /* passf4_ol_ps, passf8_ol_ps: interleaved output */
    T_NAME(PASS_KINDS), /* their number */
};

/* whether the build has the pass function of that kind and radix for m
   elements per leg and l1 groups */
static int
T_NAME(pass_exists)(int kind, int R, int m, int l1) {
    const int fused = T_FUSED;
    if (R == 8) {
        /* one element per leg; without PFFFT_TERMINAL_RADIX8, one group */
        if (m != 1 || (!PFFFT_TERMINAL_RADIX8 && l1 != 1)) {
            return 0;
        }
        return kind == T_NAME(PASS_PLAIN) || (kind == T_NAME(PASS_OL) && fused && PFFFT_TERMINAL_RADIX8);
    }
    switch (kind) {
        case T_NAME(PASS_PLAIN): return 1;
        case T_NAME(PASS_IL): return fused && l1 == 1;
        case T_NAME(PASS_IP): return fused && PFFFT_INPLACE_FIRST_PASS && l1 == 1 && m >= 2;
        case T_NAME(PASS_IL_IP): return fused && PFFFT_INPLACE_FIRST_PASS && l1 == 1;
        default: return fused && R == 4 && m == 1;
    }
}

/* the pass function of that kind and radix (pass_exists) from in to out,
   which hold nvec vectors; w: the twiddles of the legs */
static void
T_NAME(run_pass)(int kind, int R, int m, int l1, const float* in, float* out, size_t nvec, float* const* w,
                 float fsign) {
    const int idot = 2 * m;
    const T_V* x = (const T_V*)in;
    T_V* y = (T_V*)out;
    if (kind == T_NAME(PASS_IP) || kind == T_NAME(PASS_IL_IP)) {
        memcpy(out, in, nvec * T_LANES * sizeof(float));
    }
    if (kind == T_NAME(PASS_PLAIN)) {
        switch (R) {
            case 2: T_FN(passf2_ps)(idot, l1, x, y, w[0], fsign); break;
            case 3: T_FN(passf3_ps)(idot, l1, x, y, w[0], w[1], fsign); break;
            case 4: T_FN(passf4_ps)(idot, l1, x, y, w[0], w[1], w[2], fsign); break;
            case 5: T_FN(passf5_ps)(idot, l1, x, y, w[0], w[1], w[2], w[3], fsign); break;
            default: T_FN(passf8_ps)(idot, l1, x, y, fsign); break;
        }
        return;
    }
#if T_FUSED
    if (kind == T_NAME(PASS_IL)) {
        switch (R) {
            case 2: T_FN(passf2_il_ps)(idot, x, y, w[0], fsign); break;
            case 3: T_FN(passf3_il_ps)(idot, x, y, w[0], w[1], fsign); break;
            case 4: T_FN(passf4_il_ps)(idot, x, y, w[0], w[1], w[2], fsign); break;
            default: T_FN(passf5_il_ps)(idot, x, y, w[0], w[1], w[2], w[3], fsign); break;
        }
        return;
    }
    if (kind == T_NAME(PASS_OL)) {
#if PFFFT_TERMINAL_RADIX8
        if (R == 8) {
            T_FN(passf8_ol_ps)(l1, x, y, fsign);
            return;
        }
#endif
        T_FN(passf4_ol_ps)(l1, x, y, fsign);
        return;
    }
#if PFFFT_INPLACE_FIRST_PASS
    if (kind == T_NAME(PASS_IP)) {
        switch (R) {
            case 2: T_FN(passf2_ip_ps)(idot, y, w[0], fsign); break;
            case 3: T_FN(passf3_ip_ps)(idot, y, w[0], w[1], fsign); break;
            case 4: T_FN(passf4_ip_ps)(idot, y, w[0], w[1], w[2], fsign); break;
            default: T_FN(passf5_ip_ps)(idot, y, w[0], w[1], w[2], w[3], fsign); break;
        }
        return;
    }
    switch (R) {
        case 2: T_FN(passf2_il_ip_ps)(idot, y, w[0], fsign); break;
        case 3: T_FN(passf3_il_ip_ps)(idot, y, w[0], w[1], fsign); break;
        case 4: T_FN(passf4_il_ip_ps)(idot, y, w[0], w[1], w[2], fsign); break;
        default: T_FN(passf5_il_ip_ps)(idot, y, w[0], w[1], w[2], w[3], fsign); break;
    }
#endif
#endif
}

/* The two directions of a pass are mirror images: all a pass does with
   fsign is multiply imaginary parts by it, those of its twiddles and of
   its own roots of unity, so the forward pass of x is the conjugate of the
   backward pass of the conjugate of x, in every bit but the sign of a
   zero. Each pass function holds a copy of its codelet for either
   direction, without the multiplication (PF_PASS_DIRECTIONS in
   pffft_passes_impl.h): the forward copy has other operations than the
   backward one (VADD_FSIGN, VSUB_FSIGN, SVCPLXMUL_NEGI), and they have to
   round as the multiplication did. */
static int
T_NAME(test_pass_directions)(void) {
    static const int radices[] = {2, 3, 4, 5, 8};
    static const int ms[] = {1, 2, 3, 8, 15};
    static const int l1s[] = {1, 2, 5};
    const int mmax = 15, lmax = 5, V = T_LANES;
    const size_t cap = (size_t)2 * 8 * mmax * lmax * V;
    float *x = t_alloc(cap), *xc = t_alloc(cap), *f = t_alloc(cap), *b = t_alloc(cap);
    float* w[4] = {NULL, NULL, NULL, NULL};
    int fails = 0, kind, ir, im, il1, j, cases = 0;
    for (j = 0; j < 4; ++j) {
        w[j] = t_alloc((size_t)2 * mmax);
    }
    if (!x || !xc || !f || !b || !w[0] || !w[1] || !w[2] || !w[3]) {
        fails += T_CHECK(x && xc && f && b && w[0] && w[1] && w[2] && w[3]);
        goto done;
    }
    for (j = 0; j < 4; ++j) {
        t_fill(w[j], (size_t)2 * mmax);
    }
    for (kind = 0; kind < T_NAME(PASS_KINDS); ++kind) {
        for (ir = 0; ir < (int)(sizeof(radices) / sizeof(radices[0])); ++ir) {
            for (im = 0; im < (int)(sizeof(ms) / sizeof(ms[0])); ++im) {
                for (il1 = 0; il1 < (int)(sizeof(l1s) / sizeof(l1s[0])); ++il1) {
                    const int R = radices[ir], m = ms[im], l1 = l1s[il1];
                    const size_t nvec = (size_t)2 * R * m * l1, len = nvec * V;
                    const int in_il = kind == T_NAME(PASS_IL) || kind == T_NAME(PASS_IL_IP);
                    size_t i, bad = 0;
                    if (!T_NAME(pass_exists)(kind, R, m, l1)) {
                        continue;
                    }
                    t_fill(x, len);
                    T_NAME(run_pass)(kind, R, m, l1, x, f, nvec, w, -1.f);
                    memcpy(xc, x, len * sizeof(float));
                    T_NAME(conjugate)(xc, nvec, in_il);
                    T_NAME(run_pass)(kind, R, m, l1, xc, b, nvec, w, +1.f);
                    T_NAME(conjugate)(b, nvec, kind == T_NAME(PASS_OL));
                    for (i = 0; i < len; ++i) {
                        /* A zero error accepts either sign of zero. Widen before subtraction
                           so even a one-ULP difference between finite floats is detectable. */
                        bad += !t_within(fabs((double)f[i] - (double)b[i]), 0.0);
                    }
                    if (bad) {
                        printf("  (pass kind %d, radix %d, %d elements per leg, l1 %d: %d values differ between the "
                               "directions)\n",
                               kind, R, m, l1, (int)bad);
                    }
                    fails += T_CHECK(bad == 0);
                    ++cases;
                }
            }
        }
    }
    /* every radix as a plain pass at least; more with the fused layout */
    fails += T_CHECK(cases >= 5 * 3 * 4 + 1);
done:
    t_free(x);
    t_free(xc);
    t_free(f);
    t_free(b);
    for (j = 0; j < 4; ++j) {
        t_free(w[j]);
    }
    return fails;
}

#if T_FUSED
/* the routes of the complex drivers through every buffer arrangement the
   transforms use: cfftf1_il_ps with its input in work1 or in work2, and
   cfftf1_tail_ol_ps interleaving into its own input or into the other
   buffer, against the route with three distinct buffers, bit for bit; and
   the round trip. n = 2 is a single radix-2 pass with one element per leg,
   which must take the ordinary route */
static int
T_NAME(test_driver_routes)(void) {
    static const int ns[] = {
        2,  3,  4,  5,   6,   8,   10,  12,  15,  16,  20,   24,   32,   40,
        48, 64, 96, 128, 160, 192, 256, 320, 512, 960, 1024, 2048, 4096,
    };
    const int nmax = 4096, V = T_LANES;
    const size_t cap = (size_t)2 * nmax * V;
    float *wa = t_alloc(2 * nmax + 16), *x = t_alloc(cap), *w1 = t_alloc(cap), *w2 = t_alloc(cap);
    float *F = t_alloc(cap), *B = t_alloc(cap), *a = t_alloc(cap), *w = t_alloc(cap);
    float *z = t_alloc(cap), *d = t_alloc(cap);
    int ifac[IFAC_MAX_SIZE];
    int fails = 0, k;
    if (!wa || !x || !w1 || !w2 || !F || !B || !a || !w || !z || !d) {
        fails += T_CHECK(wa && x && w1 && w2 && F && B && a && w && z && d);
        goto done;
    }
    for (k = 0; k < (int)(sizeof(ns) / sizeof(ns[0])); ++k) {
        const int n = ns[k];
        const size_t len = (size_t)2 * n * V, bytes = len * sizeof(float);
        const double limit = 16 * T_EPS * n * (1 + log((double)n) / log(2.0));
        double err = 0;
        T_V* p;
        size_t i;
        int ok;
        T_FN(cffti1_ps)(n, wa, ifac);
        t_fill(x, len);
        /* forward: the reference with x, w1, w2 distinct */
        p = T_FN(cfftf1_il_ps)(n, (const T_V*)x, (T_V*)w1, (T_V*)w2, wa, ifac, -1);
        memcpy(F, p, bytes);
        /* input in work1 */
        memcpy(a, x, bytes);
        p = T_FN(cfftf1_il_ps)(n, (const T_V*)a, (T_V*)a, (T_V*)w, wa, ifac, -1);
        ok = (p == (T_V*)a || p == (T_V*)w) && memcmp(p, F, bytes) == 0;
        if (!ok) {
            printf("  (cfftf1_il_ps n = %d, input in work1: differs)\n", n);
        }
        fails += T_CHECK(ok);
        /* input in work2 */
        memcpy(a, x, bytes);
        p = T_FN(cfftf1_il_ps)(n, (const T_V*)a, (T_V*)w, (T_V*)a, wa, ifac, -1);
        ok = (p == (T_V*)a || p == (T_V*)w) && memcmp(p, F, bytes) == 0;
        if (!ok) {
            printf("  (cfftf1_il_ps n = %d, input in work2: differs)\n", n);
        }
        fails += T_CHECK(ok);
        /* backward: the reference interleaves into a third buffer */
        memcpy(d, F, bytes);
        T_FN(cfftf1_tail_ol_ps)(n, (T_V*)d, (T_V*)w, 2, 1, 0, wa, ifac, +1, (T_V*)z);
        memcpy(B, z, bytes);
        /* interleaving into its own input */
        memcpy(a, F, bytes);
        T_FN(cfftf1_tail_ol_ps)(n, (T_V*)a, (T_V*)w, 2, 1, 0, wa, ifac, +1, (T_V*)a);
        ok = memcmp(a, B, bytes) == 0;
        if (!ok) {
            printf("  (cfftf1_tail_ol_ps n = %d, into its input: differs)\n", n);
        }
        fails += T_CHECK(ok);
        /* interleaving into the other buffer */
        memcpy(a, F, bytes);
        T_FN(cfftf1_tail_ol_ps)(n, (T_V*)a, (T_V*)w, 2, 1, 0, wa, ifac, +1, (T_V*)w);
        ok = memcmp(w, B, bytes) == 0;
        if (!ok) {
            printf("  (cfftf1_tail_ol_ps n = %d, into the other buffer: differs)\n", n);
        }
        fails += T_CHECK(ok);
        /* the round trip, unscaled */
        for (i = 0; i < len; ++i) {
            err = t_max_err(err, fabs((double)B[i] - (double)n * x[i]));
        }
        if (!t_within(err, limit)) {
            printf("  (n = %d: round-trip error %g > %g)\n", n, err, limit);
        }
        fails += T_CHECK(t_within(err, limit));
    }
done:
    t_free(wa);
    t_free(x);
    t_free(w1);
    t_free(w2);
    t_free(F);
    t_free(B);
    t_free(a);
    t_free(w);
    t_free(z);
    t_free(d);
    return fails;
}
#endif

static int
T_NAME(run_backend_tests)(void) {
    int fails = 0;
    /* the macro contract: T_LANES floats in one T_V */
    fails += T_CHECK(sizeof(T_V) == T_LANES * sizeof(float));
    fails += T_NAME(test_plan_rule)();
    fails += T_NAME(test_real_odd_radix)();
#if PFFFT_TERMINAL_RADIX8
    fails += T_NAME(test_radix8_complex)();
    fails += T_NAME(test_radix8_real)();
#endif
#if PFFFT_TERMINAL_RADIX8 && T_FUSED
    fails += T_NAME(test_radix8_complex_ol)();
#endif
#if PFFFT_INPLACE_FIRST_PASS && T_FUSED
    fails += T_NAME(test_inplace_codelets)();
#endif
    fails += T_NAME(test_pass_directions)();
#if T_FUSED
    fails += T_NAME(test_driver_routes)();
#endif
    return fails;
}

#undef T_LANES
#undef T_V
#undef T_FN
#undef T_FUSED
#undef T_NAME
