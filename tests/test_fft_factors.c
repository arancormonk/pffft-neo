// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/* Modifications: Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com> */
#include "pffft/pffft.h"
#include "test_size_arg.h"
#include <limits.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static int
test_float(int TL) {
    PFFFT_Setup* S;

    for (int dir_i = 0; dir_i <= 1; ++dir_i) {
        for (int cplx_i = 0; cplx_i <= 1; ++cplx_i) {
            const pffft_transform_t cplx = (!cplx_i) ? PFFFT_REAL : PFFFT_COMPLEX;
            const int N_min = pffft_min_fft_size(cplx);
            const int N_max = N_min * 11 + N_min;
            int NTL = pffft_nearest_transform_size(TL, cplx, (!dir_i));
            // main rejects zero and negative transform sizes.
            // NOLINTNEXTLINE(clang-analyzer-optin.taint.TaintedDiv)
            double near_off = (NTL - TL) * 100.0 / (double)TL;

            fprintf(stderr,
                    "testing float, %s, %s ..\tminimum transform %d; nearest transform for %d is %d (%.2f%% off)\n",
                    (!dir_i) ? "FORWARD" : "BACKWARD", (!cplx_i) ? "REAL" : "COMPLEX", N_min, TL, NTL, near_off);

            /* step 0 when N_min == 1 (complex, SIMD_SZ==1): infinite loop */
            const int N_step = (N_min / 2) ? (N_min / 2) : 1;
            for (int N = (N_min / 2); N <= N_max; N += N_step) {
                int R = N, f2 = 0, f3 = 0, f5 = 0, tmp_f;
                const int factorizable = pffft_is_valid_size(N, cplx);
                while (R >= 5 * N_min && (R % 5) == 0) {
                    R /= 5;
                    ++f5;
                }
                while (R >= 3 * N_min && (R % 3) == 0) {
                    R /= 3;
                    ++f3;
                }
                while (R >= 2 * N_min && (R % 2) == 0) {
                    R /= 2;
                    ++f2;
                }
                tmp_f = (R == N_min) ? 1 : 0;
                if (factorizable != tmp_f) {
                    fprintf(stderr, "is_valid_size(%d) = %d, but factorization says %d\n", N, factorizable, tmp_f);
                    return 1;
                }

                S = pffft_new_setup(N, cplx);

                if (S && !factorizable) {
                    fprintf(stderr,
                            "fft setup successful, but NOT factorizable into min(=%d), 2^%d, 3^%d, 5^%d for N = %d (R "
                            "= %d)\n",
                            N_min, f2, f3, f5, N, R);
                    return 1;
                }
                if (!S && factorizable) {
                    fprintf(stderr,
                            "fft setup UNsuccessful, but factorizable into min(=%d), 2^%d, 3^%d, 5^%d for N = %d (R = "
                            "%d)\n",
                            N_min, f2, f3, f5, N, R);
                    return 1;
                }

                if (S) {
                    pffft_destroy_setup(S);
                }
            }
        }
    }
    return 0;
}

int
main(int argc, char* argv[]) {
    int N = (1 < argc) ? parse_test_size(argv[1], INT_MAX) : 2;

    if (N == 0) {
        fprintf(stderr, "Expected a positive transform size\n");
        return 2;
    }
    return test_float(N);
}
