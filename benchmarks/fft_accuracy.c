// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  fft_accuracy: the error of every transform of the pffft it is linked with
  against an exact reference, and hashes of the results, for an A/B
  comparison of two builds (tools/accuracy_compare.py).

  Cases are (kind, N) for every valid size up to --max, or the --sizes list.
  Every case gets the same inputs on every backend and in every build: uniform
  in [-1, 1) in float, from seeds --seed-base .. --seed-base + --seeds - 1,
  ceil(POINTS_PER_SEED / N) inputs per seed, each generated from a hash of
  (seed, kind, N, direction, index), so --shard and --backends never change
  them. Per backend, direction and ordering, one row:

    forward ordered     pffft_transform_ordered(x)
    forward unordered   pffft_transform(x), measured after pffft_zreorder()
    backward ordered    pffft_transform_ordered(u), u a random ordered spectrum
    backward unordered  pffft_transform(pffft_zreorder(u, PFFFT_BACKWARD))

  rms is the relative RMS error pooled over all inputs,
  sqrt(sum |y - ref|^2 / sum |ref|^2), and max the largest relative L2 error
  of one input, against the long-double DFT of fft_reference.h (unscaled
  backward). hash is a 64-bit FNV-1a hash of the bytes of every output as
  the transform wrote it, in_hash the same of the inputs it was given: for
  unordered backward those come from the build's own zreorder().

  factors lists the setup's radices in the plan's stored order (not always
  the order they run in), df its depth-first levels.

  Within the build each row must also pass, else `checks` names the failure:
    reorder    ordered results equal zreorder(unordered) forward and the
               transform of zreorder(u) backward, bit for bit
    inplace    the transform in place equals the one out of place
    nonfinite  every output is finite
    limit      no input's error exceeds TOL_FFT * eps * log2(N), the limit
               of tests/test_pffft_accuracy.c
    lanes      the setup has the lanes of its backend

  Backends (pffft_internal_new_setup()):
    native     PFFFT_BACKEND_NATIVE, pffft_simd_size() lanes
    w8         PFFFT_BACKEND_W8, the 8-lane AVX2 backend at every size it
               supports, whatever the CPU-model tables say (no rows where
               the build or the CPU lacks it)
    native-df, w8-df   forced depth-first schedule; rows only where the
               plan allows it (complex sizes with leading radix-4 passes)
    native-bf, w8-bf   forced breadth-first schedule; rows only where the
               backend's default schedule is depth first
    auto       pffft_new_setup()'s choice
  The -df and -bf backends need a library of 29493da or later.

  usage: fft_accuracy [--max N] [--sizes N,...] [--kinds real,complex]
                      [--backends LIST] [--seeds K] [--seed-base S]
                      [--shard I/K] [--hash-only]
  Writes a CSV with '#' metadata lines to stdout. Exits 1 if a check fails.
*/

#include "pffft/pffft.h"
#include "pffft_internal.h"
#include "fft_reference.h"
#include "test_size_arg.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* accepted relative L2 error of one input, in units of eps * log2(N) */
#define TOL_FFT         2.0
#define DEFAULT_MAX_N   65536
#define LARGEST_N       (1 << 22)
#define POINTS_PER_SEED 8192
#define MAX_SEEDS       64
#define TOKEN_SIZE      32

enum { SCHED_DEFAULT, SCHED_DEPTH_FIRST, SCHED_BREADTH_FIRST };

typedef struct {
    const char* name;
    int backend;  /* PFFFT_BACKEND_* without schedule flags */
    int schedule; /* SCHED_* */
    int wide;     /* 1: 8 lanes, 0: pffft_simd_size(), -1: either */
} backend_def;

static const backend_def backends[] = {
    {"native", PFFFT_BACKEND_NATIVE, SCHED_DEFAULT, 0},
    {"w8", PFFFT_BACKEND_W8, SCHED_DEFAULT, 1},
#ifdef PFFFT_BACKEND_DEPTH_FIRST
    {"native-df", PFFFT_BACKEND_NATIVE, SCHED_DEPTH_FIRST, 0},
    {"w8-df", PFFFT_BACKEND_W8, SCHED_DEPTH_FIRST, 1},
    {"native-bf", PFFFT_BACKEND_NATIVE, SCHED_BREADTH_FIRST, 0},
    {"w8-bf", PFFFT_BACKEND_W8, SCHED_BREADTH_FIRST, 1},
#endif
    {"auto", PFFFT_BACKEND_AUTO, SCHED_DEFAULT, -1},
};

#define NUM_BACKENDS ((int)(sizeof(backends) / sizeof(backends[0])))

enum { CHECK_REORDER = 1, CHECK_INPLACE = 2, CHECK_NONFINITE = 4, CHECK_LIMIT = 8, CHECK_LANES = 16 };

/* one output row: a backend, direction and ordering of one case */
typedef struct {
    ldouble num, den; /* sums of |y - ref|^2 and |ref|^2 */
    double max_err;
    unsigned long long hash, in_hash;
    int checks;
} row_acc;

typedef struct {
    int max_n;
    int* sizes; /* NULL: every valid size up to max_n */
    int num_sizes;
    int kinds[2]; /* REAL, COMPLEX selected */
    int selected[NUM_BACKENDS];
    int seeds, seed_base;
    int shard, shards;
    int hash_only;
    int w8_available; /* the probe found the 8-lane backend */
} options;

/* ---------------- hashing and inputs ---------------- */

#define FNV_OFFSET 14695981039346656037ull
#define FNV_PRIME  1099511628211ull

/* FNV-1a over the bytes of n floats, least significant byte first */
static unsigned long long
hash_floats(unsigned long long h, const float* x, int n) {
    int k, b;
    for (k = 0; k < n; ++k) {
        unsigned int bits;
        memcpy(&bits, &x[k], sizeof(bits));
        for (b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xffu;
            h *= FNV_PRIME;
        }
    }
    return h;
}

static unsigned long long
splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* input `index` of a seed, kind, size and direction */
static void
fill_input(float* x, int count, int seed, int cplx, int N, int dir, int index) {
    unsigned long long state = splitmix64((unsigned long long)seed);
    int k;
    state = splitmix64(state ^ (unsigned long long)cplx);
    state = splitmix64(state ^ (unsigned long long)N);
    state = splitmix64(state ^ (unsigned long long)dir);
    state = splitmix64(state ^ (unsigned long long)index);
    if (state == 0) {
        state = 1;
    }
    for (k = 0; k < count; ++k) {
        x[k] = (float)xorshift_urand(&state);
    }
}

/* ---------------- helpers ---------------- */

static float*
alloc_buf(int n) {
    float* buffer = pffft_aligned_malloc(sizeof(float) * (size_t)n);
    if (!buffer) {
        (void)fputs("Could not allocate FFT buffer\n", stderr);
        exit(EXIT_FAILURE);
    }
    return buffer;
}

static int
same_bits(const float* a, const float* b, int n) {
    return memcmp(a, b, sizeof(float) * (size_t)n) == 0;
}

static int
all_finite(const float* x, int n) {
    int k;
    for (k = 0; k < n; ++k) {
        if (!isfinite(x[k])) {
            return 0;
        }
    }
    return 1;
}

static double
log2N(int N) {
    return N > 1 ? log2((double)N) : 1.0;
}

/* adds the error of y (n values, complex if cplx) against ref to the row */
static void
add_error(row_acc* row, int n, const float* y, const cld* ref, int cplx, double limit) {
    ldouble num = 0, den = 0;
    double err;
    add_err_sums(n, y, ref, cplx, &num, &den);
    row->num += num;
    row->den += den;
    err = (double)(den > 0 ? sqrtl(num / den) : sqrtl(num));
    if (!(err <= row->max_err)) {
        row->max_err = err;
    }
    if (!(err <= limit)) {
        row->checks |= CHECK_LIMIT;
    }
}

/* ---------------- one case ---------------- */

/* whether the 8-lane backend supports N (pffft_wide_supports()): a
   multiple of 64 (complex) or 128 (real) with N / 8 = 2^a 3^b 5^c */
static int
w8_size(int N, int cplx) {
    int m;
    if (N % (cplx ? 64 : 128)) {
        return 0;
    }
    m = N / 8;
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

/* The setup of backend b for (kind, N), or NULL where it gives no row. Sets
   *error when a setup that must exist does not: native and auto setups at
   every valid size, 8-lane ones where the backend is available and
   supports N (else a failed allocation would go unnoticed). */
static PFFFT_Setup*
case_setup(const options* opt, const backend_def* b, int N, int cplx, int* error) {
    const pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
    const int required = b->backend != PFFFT_BACKEND_W8 || (opt->w8_available && w8_size(N, cplx));
    PFFFT_Setup* s;
    if (b->schedule == SCHED_DEFAULT) {
        s = pffft_internal_new_setup(N, tt, b->backend);
        if (!s && required) {
            *error = 1;
        }
        return s;
    }
#ifdef PFFFT_BACKEND_DEPTH_FIRST
    if (!cplx) {
        return NULL;
    }
    if (b->schedule == SCHED_DEPTH_FIRST) {
        s = pffft_internal_new_setup(N, tt, b->backend | PFFFT_BACKEND_DEPTH_FIRST);
        if (!s && required) {
            *error = 1;
        }
    } else {
        PFFFT_Setup* d = pffft_internal_new_setup(N, tt, b->backend);
        const int default_df = d && pffft_internal_depth_first_levels(d);
        if (!d && required) {
            *error = 1;
        }
        pffft_destroy_setup(d);
        s = default_df ? pffft_internal_new_setup(N, tt, b->backend | PFFFT_BACKEND_BREADTH_FIRST) : NULL;
        if (default_df && (!s || pffft_internal_depth_first_levels(s))) {
            *error = 1;
        }
        return s;
    }
    if (s && !pffft_internal_depth_first_levels(s)) {
        pffft_destroy_setup(s);
        s = NULL;
    }
    return s;
#else
    return NULL;
#endif
}

static void
print_factors(const PFFFT_Setup* s) {
#ifdef PFFFT_INTERNAL_HAS_PLAN
    int count, k;
    const int* f = pffft_internal_plan(s, &count);
    for (k = 0; k < count; ++k) {
        (void)printf("%s%d", k ? "x" : "", f[k]);
    }
#else
    (void)s;
#endif
}

static void
print_checks(int checks) {
    static const char* const names[] = {"reorder", "inplace", "nonfinite", "limit", "lanes"};
    int k, first = 1;
    for (k = 0; k < 5; ++k) {
        if (checks & (1 << k)) {
            (void)printf("%s%s", first ? "" : "+", names[k]);
            first = 0;
        }
    }
}

/* Runs one (kind, N) case on every selected backend and prints its rows.
   Returns the number of rows with a failed check, plus missing setups. */
static int
run_case(const options* opt, int cplx, int N) {
    const int nf = cplx ? 2 * N : N; /* floats per buffer */
    const int ns = cplx ? N : N / 2; /* complex slots of the ordered spectrum */
    const int per_seed = (POINTS_PER_SEED + N - 1) / N;
    const int inputs = per_seed * opt->seeds;
    const double limit = TOL_FFT * FLT_EPSILON * log2N(N);
    const int measure = !opt->hash_only;
    PFFFT_Setup* setups[NUM_BACKENDS];
    row_acc rows[NUM_BACKENDS][2][2]; /* [backend][forward, backward][ordered, unordered] */
    float *x, *y, *z, *w, *t, *work;
    cld *xc = NULL, *F = NULL, *ref = NULL;
    int failures = 0, num_setups = 0, b, dir, ord, seed, index;

    memset(rows, 0, sizeof(rows));
    for (b = 0; b < NUM_BACKENDS; ++b) {
        int error = 0;
        setups[b] = opt->selected[b] ? case_setup(opt, &backends[b], N, cplx, &error) : NULL;
        if (error) {
            (void)printf("# error: %s %s %d: no setup\n", backends[b].name, cplx ? "complex" : "real", N);
            ++failures;
        }
        num_setups += setups[b] != NULL;
        for (dir = 0; dir < 2; ++dir) {
            for (ord = 0; ord < 2; ++ord) {
                rows[b][dir][ord].hash = rows[b][dir][ord].in_hash = FNV_OFFSET;
            }
        }
        if (setups[b]) {
            const int lanes = pffft_setup_simd_size(setups[b]);
            if ((backends[b].wide == 0 && lanes != pffft_simd_size()) || (backends[b].wide == 1 && lanes != 8)) {
                rows[b][0][0].checks |= CHECK_LANES;
            }
        }
    }

    if (num_setups == 0) {
        return failures;
    }
    x = alloc_buf(nf);
    y = alloc_buf(nf);
    z = alloc_buf(nf);
    w = alloc_buf(nf);
    t = alloc_buf(nf);
    work = alloc_buf(nf);
    if (measure) {
        xc = alloc_reference(N);
        F = alloc_reference(N);
        ref = alloc_reference(N);
    }

    for (dir = 0; dir < 2; ++dir) {
        const pffft_direction_t direction = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
        for (seed = opt->seed_base; seed < opt->seed_base + opt->seeds; ++seed) {
            for (index = 0; index < per_seed; ++index) {
                const cld* expected = ref;
                int n_cmp = N, cmp_cplx = cplx;
                fill_input(x, nf, seed, cplx, N, dir, index);
                if (measure) {
                    if (!dir) {
                        to_cld(N, x, cplx, xc);
                        ref_dft(N, xc, F, -1);
                        if (cplx) {
                            expected = F;
                        } else {
                            real_spectrum_to_ordered(N, F, ref);
                            n_cmp = ns;
                            cmp_cplx = 1;
                        }
                    } else {
                        if (cplx) {
                            to_cld(N, x, 1, F);
                        } else {
                            ordered_to_real_spectrum(N, x, F);
                        }
                        ref_dft(N, F, ref, +1);
                    }
                }
                for (b = 0; b < NUM_BACKENDS; ++b) {
                    const PFFFT_Setup* s = setups[b];
                    row_acc *ro = &rows[b][dir][0], *ru = &rows[b][dir][1];
                    if (!s) {
                        continue;
                    }
                    /* ordered */
                    pffft_transform_ordered(s, x, y, work, direction);
                    ro->in_hash = hash_floats(ro->in_hash, x, nf);
                    ro->hash = hash_floats(ro->hash, y, nf);
                    memcpy(t, x, sizeof(float) * (size_t)nf);
                    pffft_transform_ordered(s, t, t, work, direction);
                    if (!same_bits(t, y, nf)) {
                        ro->checks |= CHECK_INPLACE;
                    }
                    if (!all_finite(y, nf)) {
                        ro->checks |= CHECK_NONFINITE;
                    }
                    if (measure) {
                        add_error(ro, n_cmp, y, expected, cmp_cplx, limit);
                    }
                    /* unordered: z the result, w its reordered form or the reordered input */
                    if (!dir) {
                        pffft_transform(s, x, z, work, direction);
                        ru->in_hash = hash_floats(ru->in_hash, x, nf);
                        memcpy(t, x, sizeof(float) * (size_t)nf);
                        pffft_zreorder(s, z, w, PFFFT_FORWARD);
                        if (!same_bits(w, y, nf)) {
                            ru->checks |= CHECK_REORDER;
                        }
                    } else {
                        pffft_zreorder(s, x, w, PFFFT_BACKWARD);
                        pffft_transform(s, w, z, work, direction);
                        ru->in_hash = hash_floats(ru->in_hash, w, nf);
                        memcpy(t, w, sizeof(float) * (size_t)nf);
                        if (!same_bits(z, y, nf)) {
                            ru->checks |= CHECK_REORDER;
                        }
                    }
                    ru->hash = hash_floats(ru->hash, z, nf);
                    pffft_transform(s, t, t, work, direction);
                    if (!same_bits(t, z, nf)) {
                        ru->checks |= CHECK_INPLACE;
                    }
                    if (!all_finite(z, nf)) {
                        ru->checks |= CHECK_NONFINITE;
                    }
                    if (measure) {
                        add_error(ru, n_cmp, dir ? z : w, expected, cmp_cplx, limit);
                    }
                }
            }
        }
    }

    for (b = 0; b < NUM_BACKENDS; ++b) {
        if (!setups[b]) {
            continue;
        }
        if (rows[b][0][0].checks & CHECK_LANES) {
            for (dir = 0; dir < 2; ++dir) {
                rows[b][dir][0].checks |= CHECK_LANES;
                rows[b][dir][1].checks |= CHECK_LANES;
            }
        }
        for (dir = 0; dir < 2; ++dir) {
            for (ord = 0; ord < 2; ++ord) {
                const row_acc* r = &rows[b][dir][ord];
                (void)printf("%s,%d,%s,%d,%s,%s,", backends[b].name, pffft_setup_simd_size(setups[b]),
                             cplx ? "complex" : "real", N, dir ? "backward" : "forward", ord ? "unordered" : "ordered");
                print_factors(setups[b]);
#ifdef PFFFT_BACKEND_DEPTH_FIRST
                (void)printf(",%d,%d,", pffft_internal_depth_first_levels(setups[b]), inputs);
#else
                (void)printf(",,%d,", inputs);
#endif
                if (measure) {
                    const double rms = (double)(r->den > 0 ? sqrtl(r->num / r->den) : sqrtl(r->num));
                    (void)printf("%.9e,%.9e", rms, r->max_err);
                } else {
                    (void)printf(",");
                }
                (void)printf(",%016llx,%016llx,", r->in_hash, r->hash);
                print_checks(r->checks);
                (void)printf("\n");
                failures += r->checks != 0;
            }
        }
        pffft_destroy_setup(setups[b]);
    }
    (void)fflush(stdout);

    pffft_aligned_free(x);
    pffft_aligned_free(y);
    pffft_aligned_free(z);
    pffft_aligned_free(w);
    pffft_aligned_free(t);
    pffft_aligned_free(work);
    free(xc);
    free(F);
    free(ref);
    return failures;
}

/* ---------------- command line ---------------- */

static int
usage(const char* argv0) {
    (void)fprintf(stderr,
                  "usage: %s [--max N] [--sizes N,...] [--kinds real,complex] [--backends LIST]\n"
                  "       [--seeds K] [--seed-base S] [--shard I/K] [--hash-only]\n"
                  "LIST: comma-separated, of",
                  argv0);
    for (int b = 0; b < NUM_BACKENDS; ++b) {
        (void)fprintf(stderr, " %s", backends[b].name);
    }
    (void)fprintf(stderr, " (default: all)\n");
    return 2;
}

/* Copies the next comma-separated token of *list to token and advances
   *list past it. Returns 0 at the end of the list or for an empty or too
   long token. */
static int
next_token(const char** list, char* token) {
    const char* start = *list;
    const char* end = strchr(start, ',');
    size_t len = end ? (size_t)(end - start) : strlen(start);
    if (*start == '\0' || len == 0 || len >= TOKEN_SIZE) {
        return 0;
    }
    memcpy(token, start, len);
    token[len] = '\0';
    *list = end ? end + 1 : start + len;
    return 1;
}

/* the number of tokens of a comma-separated list, 0 if one is malformed */
static int
count_tokens(const char* list) {
    char token[TOKEN_SIZE];
    int count = 0;
    while (*list) {
        if (!next_token(&list, token)) {
            return 0;
        }
        ++count;
    }
    return count;
}

static int
parse_backends(const char* list, options* opt) {
    char token[TOKEN_SIZE];
    int b;
    if (count_tokens(list) == 0) {
        return 0;
    }
    memset(opt->selected, 0, sizeof(opt->selected));
    while (*list) {
        int found = 0;
        if (!next_token(&list, token)) {
            return 0;
        }
        for (b = 0; b < NUM_BACKENDS; ++b) {
            if (strcmp(token, backends[b].name) == 0) {
                opt->selected[b] = found = 1;
            }
        }
        if (!found) {
            (void)fprintf(stderr, "unknown or unsupported backend: %s\n", token);
            return 0;
        }
    }
    return 1;
}

static int
parse_kinds(const char* list, options* opt) {
    char token[TOKEN_SIZE];
    if (count_tokens(list) == 0) {
        return 0;
    }
    opt->kinds[0] = opt->kinds[1] = 0;
    while (*list) {
        if (!next_token(&list, token)) {
            return 0;
        }
        if (strcmp(token, "real") == 0) {
            opt->kinds[0] = 1;
        } else if (strcmp(token, "complex") == 0) {
            opt->kinds[1] = 1;
        } else {
            return 0;
        }
    }
    return 1;
}

static int
parse_sizes(const char* list, options* opt) {
    char token[TOKEN_SIZE];
    const int count = count_tokens(list);
    int k;
    if (count == 0) {
        return 0;
    }
    free(opt->sizes);
    opt->sizes = malloc(sizeof(int) * (size_t)count);
    if (!opt->sizes) {
        return 0;
    }
    opt->num_sizes = count;
    for (k = 0; k < count; ++k) {
        if (!next_token(&list, token)) {
            return 0;
        }
        opt->sizes[k] = parse_test_size(token, LARGEST_N);
        if (opt->sizes[k] == 0) {
            return 0;
        }
    }
    return 1;
}

/* I/K with 0 <= I < K */
static int
parse_shard(const char* text, options* opt) {
    char token[TOKEN_SIZE];
    const char* slash = strchr(text, '/');
    size_t len = slash ? (size_t)(slash - text) : 0;
    int index;
    if (!slash || len == 0 || len >= TOKEN_SIZE) {
        return 0;
    }
    memcpy(token, text, len);
    token[len] = '\0';
    /* parse_test_size() accepts positive values only */
    if (strcmp(token, "0") == 0) {
        index = 0;
    } else {
        index = parse_test_size(token, 1 << 20);
        if (index == 0) {
            return 0;
        }
    }
    opt->shards = parse_test_size(slash + 1, 1 << 20);
    opt->shard = index;
    return opt->shards > 0 && index < opt->shards;
}

static int
parse_args(int argc, char** argv, options* opt) {
    int i, b;
    memset(opt, 0, sizeof(*opt));
    opt->max_n = DEFAULT_MAX_N;
    opt->kinds[0] = opt->kinds[1] = 1;
    opt->seeds = 2;
    opt->seed_base = 1;
    opt->shards = 1;
    for (b = 0; b < NUM_BACKENDS; ++b) {
        opt->selected[b] = 1;
    }
    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* value = i + 1 < argc ? argv[i + 1] : NULL;
        int ok;
        if (strcmp(arg, "--hash-only") == 0) {
            opt->hash_only = 1;
            continue;
        }
        if (!value) {
            return 0;
        }
        if (strcmp(arg, "--max") == 0) {
            ok = (opt->max_n = parse_test_size(value, LARGEST_N)) != 0;
        } else if (strcmp(arg, "--sizes") == 0) {
            ok = parse_sizes(value, opt);
        } else if (strcmp(arg, "--kinds") == 0) {
            ok = parse_kinds(value, opt);
        } else if (strcmp(arg, "--backends") == 0) {
            ok = parse_backends(value, opt);
        } else if (strcmp(arg, "--seeds") == 0) {
            ok = (opt->seeds = parse_test_size(value, MAX_SEEDS)) != 0;
        } else if (strcmp(arg, "--seed-base") == 0) {
            ok = (opt->seed_base = parse_test_size(value, 1 << 20)) != 0;
        } else if (strcmp(arg, "--shard") == 0) {
            ok = parse_shard(value, opt);
        } else {
            ok = 0;
        }
        if (!ok) {
            (void)fprintf(stderr, "invalid argument: %s %s\n", arg, value);
            return 0;
        }
        ++i;
    }
    return 1;
}

/* whether (kind, N) is case number `index` of this shard's */
static int
in_shard(const options* opt, int index) {
    return index % opt->shards == opt->shard;
}

int
main(int argc, char** argv) {
    options opt;
    int cplx, k, index = 0, failures = 0, cases = 0;

    if (!parse_args(argc, argv, &opt)) {
        free(opt.sizes);
        return usage(argv[0]);
    }
    {
        PFFFT_Setup* probe = pffft_internal_new_setup(128, PFFFT_REAL, PFFFT_BACKEND_W8);
        opt.w8_available = probe != NULL;
        pffft_destroy_setup(probe);
    }
    (void)printf("# fft_accuracy 1\n# simd_arch=%s\n# simd_size=%d\n# w8=%d\n# ldbl_mant_dig=%d\n", pffft_simd_arch(),
                 pffft_simd_size(), opt.w8_available, LDBL_MANT_DIG);
    (void)printf("# seeds=%d..%d\n# points_per_seed=%d\n# hash_only=%d\n", opt.seed_base, opt.seed_base + opt.seeds - 1,
                 POINTS_PER_SEED, opt.hash_only);
    if (!opt.hash_only) {
        unsigned long long state = 0x9E3779B97F4A7C15ull;
        const double worst = reference_self_check(&state);
        (void)printf("# reference_self_check=%.3g\n", worst);
        if (!(worst <= 64 * LDBL_EPSILON * log2(200.0))) {
            (void)printf("# error: the reference DFT disagrees with the naive DFT\n");
            ++failures;
        }
    }
    (void)printf("backend,lanes,kind,n,direction,order,factors,df,inputs,rms,max,in_hash,hash,checks\n");

    for (cplx = 0; cplx < 2; ++cplx) {
        const pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
        const int nmin = pffft_min_fft_size(tt);
        if (!opt.kinds[cplx]) {
            continue;
        }
        if (opt.sizes) {
            for (k = 0; k < opt.num_sizes; ++k) {
                if (pffft_is_valid_size(opt.sizes[k], tt) && in_shard(&opt, index++)) {
                    failures += run_case(&opt, cplx, opt.sizes[k]);
                    ++cases;
                }
            }
        } else {
            for (k = nmin; k <= opt.max_n; k += nmin) {
                if (pffft_is_valid_size(k, tt) && in_shard(&opt, index++)) {
                    failures += run_case(&opt, cplx, k);
                    ++cases;
                }
            }
        }
    }
    release_twiddles();
    free(opt.sizes);
    (void)printf("# end cases=%d failures=%d\n", cases, failures);
    if (fflush(stdout) != 0 || ferror(stdout)) {
        return 2;
    }
    return failures ? 1 : 0;
}
