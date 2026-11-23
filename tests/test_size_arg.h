// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#ifndef PFFFT_TEST_SIZE_ARG_H
#define PFFFT_TEST_SIZE_ARG_H

#include <errno.h>
#include <stdlib.h>

/* Return zero for an invalid size, including a value outside the test range. */
static int
parse_test_size(const char* text, int upper) {
    char* end;
    errno = 0;
    long size = strtol(text, &end, 10);
    return end != text && *end == '\0' && errno == 0 && size > 0 && size <= upper ? (int)size : 0;
}
#endif
