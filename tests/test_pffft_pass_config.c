// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Tests of src/pffft_pass_config.h: what it leaves in PFFFT_TERMINAL_RADIX8
  and PFFFT_INPLACE_FIRST_PASS for the defaults and for definitions made
  before it, among them expressions without parentheses such as 0 ? 0 : 1.
  PFFFT_DEPTH_FIRST, normalized the same way, has three cases of its own,
  and the 4-lane depth-first thresholds by vendor five, at the end.

  The header is included once per case, each time with the definitions of
  that case (the include guard, the two switches and SIMD_SZ undefined in
  between). Right after each inclusion the switches are captured in a
  t_case: their expansion as text, which must be the literal 0 or 1, and
  expressions whose value depends on how the expansion groups: 2 * X (2
  for the literal 1, 1 for 0 ? 0 : 1 and for 1 - 1), X - 1 (-1 for the
  literal 0, 0 for 0 && 1), -X (-1 for the literal 1, 1 for 0 ? 0 : 1),
  X && 0 (0 for the literal, 1 for 1 || 0), and whether a block
  #if X && 0 was compiled. The definitions of the build, if any, are
  undefined before the first case.

  Needs no library. Prints one line per failing check and exits with a
  non-zero status if any check failed.
*/

#include <stdio.h>
#include <string.h>

#define T_STR_(x) #x
#define T_STR(x)  T_STR_(x)

/* the switches after one inclusion of the header, and their values */
typedef struct {
    const char* what;              /* the case */
    int r8, ip;                    /* the values the switches must have */
    const char *r8_text, *ip_text; /* their expansions */
    int r8_twice, ip_twice;        /* 2 * X */
    int r8_less1, ip_less1;        /* X - 1 */
    int r8_neg, ip_neg;            /* -X */
    int r8_and0, ip_and0;          /* X && 0 */
    int r8_if_and0, ip_if_and0;    /* 1 if a block #if X && 0 was compiled */
} t_case;

/* a t_case of the switches as the header left them; r8 and ip are the
   values they must have. T_IF_R8_AND0 and T_IF_IP_AND0 are set by the
   block of the case. */
#define T_CAPTURE(what, r8, ip)                                                                                        \
    {what,                                                                                                             \
     r8,                                                                                                               \
     ip,                                                                                                               \
     T_STR(PFFFT_TERMINAL_RADIX8),                                                                                     \
     T_STR(PFFFT_INPLACE_FIRST_PASS),                                                                                  \
     2 * PFFFT_TERMINAL_RADIX8,                                                                                        \
     2 * PFFFT_INPLACE_FIRST_PASS,                                                                                     \
     PFFFT_TERMINAL_RADIX8 - 1,                                                                                        \
     PFFFT_INPLACE_FIRST_PASS - 1,                                                                                     \
     -PFFFT_TERMINAL_RADIX8,                                                                                           \
     -PFFFT_INPLACE_FIRST_PASS,                                                                                        \
     PFFFT_TERMINAL_RADIX8 && 0,                                                                                       \
     PFFFT_INPLACE_FIRST_PASS && 0,                                                                                    \
     T_IF_R8_AND0,                                                                                                     \
     T_IF_IP_AND0}

/* 1: no definition, 4 lanes: both 1 */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#define SIMD_SZ 4
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_1 = T_CAPTURE("defaults, SIMD_SZ 4", 1, 1);

/* 2: no definition, single lane: the terminal radix-8 pass is off */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_2 = T_CAPTURE("defaults, SIMD_SZ 1", 0, 1);

/* 3: expressions of value 1 without parentheses */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  4
#define PFFFT_TERMINAL_RADIX8    0 ? 0 : 1
#define PFFFT_INPLACE_FIRST_PASS 1 || 0
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_3 = T_CAPTURE("0 ? 0 : 1 and 1 || 0, SIMD_SZ 4", 1, 1);

/* 4: the same, the other way round, single lane (a definition overrides
   the default of SIMD_SZ 1) */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  1
#define PFFFT_TERMINAL_RADIX8    1 || 0
#define PFFFT_INPLACE_FIRST_PASS 0 ? 0 : 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_4 = T_CAPTURE("1 || 0 and 0 ? 0 : 1, SIMD_SZ 1", 1, 1);

/* 5: expressions of value 0 without parentheses */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  4
// NOLINTNEXTLINE(bugprone-macro-parentheses) -- deliberately unparenthesized build definition.
#define PFFFT_TERMINAL_RADIX8    1 - 1
#define PFFFT_INPLACE_FIRST_PASS 0 && 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_5 = T_CAPTURE("1 - 1 and 0 && 1, SIMD_SZ 4", 0, 0);

/* 6: the same, the other way round */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  4
#define PFFFT_TERMINAL_RADIX8    0 && 1
// NOLINTNEXTLINE(bugprone-macro-parentheses) -- deliberately unparenthesized build definition.
#define PFFFT_INPLACE_FIRST_PASS 1 - 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_6 = T_CAPTURE("0 && 1 and 1 - 1, SIMD_SZ 4", 0, 0);

/* 7: parenthesised */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  4
#define PFFFT_TERMINAL_RADIX8    (1)
#define PFFFT_INPLACE_FIRST_PASS (0)
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_7 = T_CAPTURE("(1) and (0), SIMD_SZ 4", 1, 0);

/* 8: parenthesised, the other way round, single lane */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  1
#define PFFFT_TERMINAL_RADIX8    (0)
#define PFFFT_INPLACE_FIRST_PASS (1)
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_8 = T_CAPTURE("(0) and (1), SIMD_SZ 1", 0, 1);

/* 9: plain literals (as CMake defines them for ON and OFF) */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  4
#define PFFFT_TERMINAL_RADIX8    1
#define PFFFT_INPLACE_FIRST_PASS 0
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_9 = T_CAPTURE("1 and 0, SIMD_SZ 4", 1, 0);

/* 10: plain literals, the other way round, single lane */
#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_TERMINAL_RADIX8
#undef PFFFT_INPLACE_FIRST_PASS
#undef SIMD_SZ
#undef T_IF_R8_AND0
#undef T_IF_IP_AND0
#define SIMD_SZ                  1
#define PFFFT_TERMINAL_RADIX8    0
#define PFFFT_INPLACE_FIRST_PASS 1
#include "pffft_pass_config.h"
#if PFFFT_TERMINAL_RADIX8 && 0
#define T_IF_R8_AND0 1
#else
#define T_IF_R8_AND0 0
#endif
#if PFFFT_INPLACE_FIRST_PASS && 0
#define T_IF_IP_AND0 1
#else
#define T_IF_IP_AND0 0
#endif
static const t_case t_case_10 = T_CAPTURE("0 and 1, SIMD_SZ 1", 0, 1);

/* PFFFT_DEPTH_FIRST: the default, then expressions of value 1 and 0
   without parentheses; the same captures as above */
typedef struct {
    const char* what;
    int df;           /* the value the switch must have */
    const char* text; /* its expansion */
    int twice, less1, neg, and0, if_and0;
} t_df_case;

#define T_DF_CAPTURE(what, df)                                                                                         \
    {what,                                                                                                             \
     df,                                                                                                               \
     T_STR(PFFFT_DEPTH_FIRST),                                                                                         \
     2 * PFFFT_DEPTH_FIRST,                                                                                            \
     PFFFT_DEPTH_FIRST - 1,                                                                                            \
     -PFFFT_DEPTH_FIRST,                                                                                               \
     PFFFT_DEPTH_FIRST && 0,                                                                                           \
     T_IF_DF_AND0}

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST
#include "pffft_pass_config.h"
#if PFFFT_DEPTH_FIRST && 0
#define T_IF_DF_AND0 1
#else
#define T_IF_DF_AND0 0
#endif
static const t_df_case t_df_case_1 = T_DF_CAPTURE("PFFFT_DEPTH_FIRST, default", 1);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST
#undef T_IF_DF_AND0
#define PFFFT_DEPTH_FIRST 0 ? 0 : 1
#include "pffft_pass_config.h"
#if PFFFT_DEPTH_FIRST && 0
#define T_IF_DF_AND0 1
#else
#define T_IF_DF_AND0 0
#endif
static const t_df_case t_df_case_2 = T_DF_CAPTURE("PFFFT_DEPTH_FIRST 0 ? 0 : 1", 1);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST
#undef T_IF_DF_AND0
// NOLINTNEXTLINE(bugprone-macro-parentheses) -- deliberately unparenthesized build definition.
#define PFFFT_DEPTH_FIRST 1 - 1
#include "pffft_pass_config.h"
#if PFFFT_DEPTH_FIRST && 0
#define T_IF_DF_AND0 1
#else
#define T_IF_DF_AND0 0
#endif
static const t_df_case t_df_case_3 = T_DF_CAPTURE("PFFFT_DEPTH_FIRST 1 - 1", 0);

static const t_df_case* const t_df_cases[] = {&t_df_case_1, &t_df_case_2, &t_df_case_3};

/* The 4-lane depth-first thresholds: the defaults by vendor, a build's
   PFFFT_DEPTH_FIRST_MIN_BYTES for every vendor, a vendor's own value over
   it, and a vendor's own value alone. 2 * X shows how the thresholds that
   the header derives from a definition without parentheses group. */
typedef struct {
    const char* what;
    long long other, intel, amd;             /* the values they must have */
    long long got_other, got_intel, got_amd; /* their values */
    long long intel_twice, amd_twice;        /* 2 * X */
} t_thr_case;

#define T_THR_CAPTURE(what, other, intel, amd)                                                                         \
    {what,                                                                                                             \
     other,                                                                                                            \
     intel,                                                                                                            \
     amd,                                                                                                              \
     (long long)(PFFFT_DEPTH_FIRST_MIN_BYTES),                                                                         \
     (long long)(PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL),                                                                   \
     (long long)(PFFFT_DEPTH_FIRST_MIN_BYTES_AMD),                                                                     \
     2 * PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL,                                                                            \
     2 * PFFFT_DEPTH_FIRST_MIN_BYTES_AMD}

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST_MIN_BYTES
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE
#include "pffft_pass_config.h"
static const t_thr_case t_thr_case_1 = T_THR_CAPTURE("thresholds, default", 8LL << 20, 512LL << 10, 16LL << 20);
/* and the 8-lane one */
static const long long t_wide_default = (long long)(PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST_MIN_BYTES
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#define PFFFT_DEPTH_FIRST_MIN_BYTES 0
#include "pffft_pass_config.h"
static const t_thr_case t_thr_case_2 = T_THR_CAPTURE("thresholds, PFFFT_DEPTH_FIRST_MIN_BYTES 0", 0, 0, 0);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST_MIN_BYTES
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
// NOLINTNEXTLINE(bugprone-macro-parentheses) -- deliberately unparenthesized build definition.
#define PFFFT_DEPTH_FIRST_MIN_BYTES 4096 + 4096
#include "pffft_pass_config.h"
static const t_thr_case t_thr_case_3 =
    T_THR_CAPTURE("thresholds, PFFFT_DEPTH_FIRST_MIN_BYTES 4096 + 4096", 8192, 8192, 8192);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST_MIN_BYTES
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#define PFFFT_DEPTH_FIRST_MIN_BYTES       0
#define PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL 2048
#include "pffft_pass_config.h"
static const t_thr_case t_thr_case_4 =
    T_THR_CAPTURE("thresholds, PFFFT_DEPTH_FIRST_MIN_BYTES 0, ..._INTEL 2048", 0, 2048, 0);

#undef PFFFT_PASS_CONFIG_H
#undef PFFFT_DEPTH_FIRST_MIN_BYTES
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL
#undef PFFFT_DEPTH_FIRST_MIN_BYTES_AMD
#define PFFFT_DEPTH_FIRST_MIN_BYTES_AMD 4096
#include "pffft_pass_config.h"
static const t_thr_case t_thr_case_5 =
    T_THR_CAPTURE("thresholds, PFFFT_DEPTH_FIRST_MIN_BYTES_AMD 4096", 8LL << 20, 512LL << 10, 4096);

static const t_thr_case* const t_thr_cases[] = {&t_thr_case_1, &t_thr_case_2, &t_thr_case_3, &t_thr_case_4,
                                                &t_thr_case_5};

static const t_case* const t_cases[] = {
    &t_case_1, &t_case_2, &t_case_3, &t_case_4, &t_case_5, &t_case_6, &t_case_7, &t_case_8, &t_case_9, &t_case_10,
};

/* 1 and a report line when got != expected, else 0 */
static int
t_int(const char* what, const char* check, int got, int expected) {
    if (got == expected) {
        return 0;
    }
    printf("FAIL %s: %s is %d, expected %d\n", what, check, got, expected);
    return 1;
}

/* 1 and a report line when the expansion is not the literal of value,
   else 0 */
static int
t_text(const char* what, const char* check, const char* got, int value) {
    const char* expected = value ? "1" : "0";
    if (strcmp(got, expected) == 0) {
        return 0;
    }
    printf("FAIL %s: %s expands to \"%s\", expected \"%s\"\n", what, check, got, expected);
    return 1;
}

static int
t_check_case(const t_case* c) {
    int fails = 0;
    fails += t_text(c->what, "PFFFT_TERMINAL_RADIX8", c->r8_text, c->r8);
    fails += t_text(c->what, "PFFFT_INPLACE_FIRST_PASS", c->ip_text, c->ip);
    fails += t_int(c->what, "2 * PFFFT_TERMINAL_RADIX8", c->r8_twice, 2 * c->r8);
    fails += t_int(c->what, "2 * PFFFT_INPLACE_FIRST_PASS", c->ip_twice, 2 * c->ip);
    fails += t_int(c->what, "PFFFT_TERMINAL_RADIX8 - 1", c->r8_less1, c->r8 - 1);
    fails += t_int(c->what, "PFFFT_INPLACE_FIRST_PASS - 1", c->ip_less1, c->ip - 1);
    fails += t_int(c->what, "-PFFFT_TERMINAL_RADIX8", c->r8_neg, -c->r8);
    fails += t_int(c->what, "-PFFFT_INPLACE_FIRST_PASS", c->ip_neg, -c->ip);
    fails += t_int(c->what, "PFFFT_TERMINAL_RADIX8 && 0", c->r8_and0, 0);
    fails += t_int(c->what, "PFFFT_INPLACE_FIRST_PASS && 0", c->ip_and0, 0);
    fails += t_int(c->what, "#if PFFFT_TERMINAL_RADIX8 && 0 compiled", c->r8_if_and0, 0);
    fails += t_int(c->what, "#if PFFFT_INPLACE_FIRST_PASS && 0 compiled", c->ip_if_and0, 0);
    return fails;
}

static int
t_check_df_case(const t_df_case* c) {
    int fails = 0;
    fails += t_text(c->what, "PFFFT_DEPTH_FIRST", c->text, c->df);
    fails += t_int(c->what, "2 * PFFFT_DEPTH_FIRST", c->twice, 2 * c->df);
    fails += t_int(c->what, "PFFFT_DEPTH_FIRST - 1", c->less1, c->df - 1);
    fails += t_int(c->what, "-PFFFT_DEPTH_FIRST", c->neg, -c->df);
    fails += t_int(c->what, "PFFFT_DEPTH_FIRST && 0", c->and0, 0);
    fails += t_int(c->what, "#if PFFFT_DEPTH_FIRST && 0 compiled", c->if_and0, 0);
    return fails;
}

/* 1 and a report line when got != expected, else 0 */
static int
t_ll(const char* what, const char* check, long long got, long long expected) {
    if (got == expected) {
        return 0;
    }
    printf("FAIL %s: %s is %lld, expected %lld\n", what, check, got, expected);
    return 1;
}

static int
t_check_thr_case(const t_thr_case* c) {
    int fails = 0;
    fails += t_ll(c->what, "PFFFT_DEPTH_FIRST_MIN_BYTES", c->got_other, c->other);
    fails += t_ll(c->what, "PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL", c->got_intel, c->intel);
    fails += t_ll(c->what, "PFFFT_DEPTH_FIRST_MIN_BYTES_AMD", c->got_amd, c->amd);
    fails += t_ll(c->what, "2 * PFFFT_DEPTH_FIRST_MIN_BYTES_INTEL", c->intel_twice, 2 * c->intel);
    fails += t_ll(c->what, "2 * PFFFT_DEPTH_FIRST_MIN_BYTES_AMD", c->amd_twice, 2 * c->amd);
    return fails;
}

int
main(void) {
    int fails = 0;
    size_t i;
    for (i = 0; i < sizeof(t_cases) / sizeof(t_cases[0]); ++i) {
        fails += t_check_case(t_cases[i]);
    }
    for (i = 0; i < sizeof(t_df_cases) / sizeof(t_df_cases[0]); ++i) {
        fails += t_check_df_case(t_df_cases[i]);
    }
    fails += t_ll("thresholds, default", "PFFFT_DEPTH_FIRST_MIN_BYTES_WIDE", t_wide_default, 32LL << 10);
    for (i = 0; i < sizeof(t_thr_cases) / sizeof(t_thr_cases[0]); ++i) {
        fails += t_check_thr_case(t_thr_cases[i]);
    }
    if (fails) {
        printf("%d check(s) failed\n", fails);
        return 1;
    }
    printf("all checks passed (%d cases)\n",
           (int)(sizeof(t_cases) / sizeof(t_cases[0]) + sizeof(t_df_cases) / sizeof(t_df_cases[0])
                 + sizeof(t_thr_cases) / sizeof(t_thr_cases[0])));
    return 0;
}
