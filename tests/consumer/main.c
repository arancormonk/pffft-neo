// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#include <pffft/pffft.h>
#include <stdlib.h>

int
main(void) {
    enum { size = 64, count = 2 * size };

    PFFFT_Setup* setup = pffft_new_setup(size, PFFFT_COMPLEX);
    float* data = pffft_aligned_malloc(count * sizeof(*data));
    int status = EXIT_FAILURE;
    if (setup && data) {
        data[0] = 1.0f;
        for (size_t i = 1; i < count; ++i) {
            data[i] = 0.0f;
        }
        pffft_transform_ordered(setup, data, data, NULL, PFFFT_FORWARD);
        status = EXIT_SUCCESS;
        for (size_t bin = 0; bin < size; ++bin) {
            if (data[2 * bin] != 1.0f || data[2 * bin + 1] != 0.0f) {
                status = EXIT_FAILURE;
                break;
            }
        }
    }
    pffft_aligned_free(data);
    if (setup) {
        pffft_destroy_setup(setup);
    }
    return status;
}
