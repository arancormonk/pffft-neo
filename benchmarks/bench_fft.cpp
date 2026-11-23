// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
// Float FFT driver adapted from the PFFFT fork's bench_hotloops.cpp.
// Allocation, round-trip validation and checksums stay outside the timed loops.
#include "pffft/pffft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using perf_clock = std::chrono::steady_clock;
using Buffer = std::unique_ptr<float[], decltype(&pffft_aligned_free)>;

Buffer
make_buffer(int count) {
    Buffer buffer(static_cast<float*>(pffft_aligned_malloc(count * sizeof(float))), pffft_aligned_free);
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }
    std::fill_n(buffer.get(), count, 0.0f);
    return buffer;
}

double
elapsed(perf_clock::time_point start) {
    return std::chrono::duration<double>(perf_clock::now() - start).count();
}

template <class Function>
void
measure(const std::string& backend, const char* kind, const std::string& op, int n, int repeats, double seconds,
        int output_count, float* output, Function run) {
    size_t count = 1;
    double calibration;
    do {
        const auto start = perf_clock::now();
        for (size_t i = 0; i < count; ++i) {
            run();
        }
        calibration = elapsed(start);
        if (calibration < 0.01) {
            count *= 2;
        }
    } while (calibration < 0.01);
    count = std::max<size_t>(1, static_cast<size_t>(count * seconds / calibration));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::memset(output, 0, output_count * sizeof(float));
        const auto start = perf_clock::now();
        for (size_t i = 0; i < count; ++i) {
            run();
        }
        const double duration = elapsed(start);
        double checksum = 0;
        for (int i = 0; i < output_count; ++i) {
            if (!std::isfinite(output[i])) {
                throw std::runtime_error("nonfinite output");
            }
            checksum += output[i];
        }
        if (std::printf("float,%s,%s,%s,%d,%d,%zu,%.9f,%.3f,%.9g\n", backend.c_str(), kind, op.c_str(), n, repeat,
                        count, duration, duration * 1e9 / count, checksum)
                < 0
            || std::fflush(stdout) != 0) {
            throw std::runtime_error("could not write benchmark sample");
        }
    }
}

void
fill(float* x, int n) {
    unsigned state = 1729;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        x[i] = static_cast<float>(static_cast<int>(state >> 16) - 32768) / 65536;
    }
}

void
check_equal(const float* a, const float* b, int n, double scale) {
    const double tolerance = 2e-5;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || std::fabs(a[i] * scale - b[i]) > tolerance) {
            (void)std::fprintf(stderr, "validation failed at %d: %.17g vs %.17g\n", i, a[i] * scale,
                               static_cast<double>(b[i]));
            throw std::runtime_error("FFT validation failed");
        }
    }
}

} // namespace

int
main(int argc, char** argv) {
    if (argc != 6) {
        (void)std::fprintf(stderr,
                           "usage: %s real|complex OP N REPEATS SECONDS\n"
                           "OP: forward backward ordered-forward ordered-backward reorder "
                           "inplace-forward inplace-backward inplace-ordered-forward "
                           "inplace-ordered-backward zconv "
                           "pipeline\n",
                           argv[0]);
        return 2;
    }
    try {
        const std::string kind(argv[1]), op(argv[2]);
        std::size_t n_end, repeats_end, seconds_end;
        const int n = std::stoi(argv[3], &n_end), repeats = std::stoi(argv[4], &repeats_end);
        const double seconds = std::stod(argv[5], &seconds_end);
        if (argv[3][n_end] != '\0' || argv[4][repeats_end] != '\0' || argv[5][seconds_end] != '\0') {
            throw std::runtime_error("invalid numeric argument");
        }
        const bool complex = kind == "complex";
        if ((kind != "real" && !complex) || n < 1 || n > (1 << 22)
            || pffft_is_valid_size(n, complex ? PFFFT_COMPLEX : PFFFT_REAL) == 0 || repeats < 1 || repeats > 100
            || !std::isfinite(seconds) || seconds <= 0 || seconds > 60) {
            throw std::runtime_error("invalid arguments: KIND is real or complex, N a size the library "
                                     "supports up to 2^22, REPEATS 1 to 100, SECONDS above 0 up to 60");
        }
        const int count = complex ? 2 * n : n;
        // Allocate buffers before creating the setup. Their page offsets affect FFT timings.
        auto input = make_buffer(count);
        auto spectrum = make_buffer(count);
        auto filter = make_buffer(count);
        auto product = make_buffer(count);
        auto output = make_buffer(count);
        auto work = make_buffer(count);
        auto ordered = make_buffer(count);

        fill(input.get(), count);

        const std::unique_ptr<PFFFT_Setup, decltype(&pffft_destroy_setup)> plan(
            pffft_new_setup(n, complex ? PFFFT_COMPLEX : PFFFT_REAL), pffft_destroy_setup);
        PFFFT_Setup* setup = plan.get();
        if (setup == nullptr) {
            throw std::runtime_error("invalid arguments: no FFT setup for this KIND and N");
        }
        const std::string backend =
            std::string(pffft_simd_arch()) + "-x" + std::to_string(pffft_setup_simd_size(setup));
        pffft_transform(setup, input.get(), spectrum.get(), work.get(), PFFFT_FORWARD);
        pffft_transform(setup, spectrum.get(), output.get(), work.get(), PFFFT_BACKWARD);
        check_equal(output.get(), input.get(), count, 1.0 / n);
        pffft_transform_ordered(setup, input.get(), ordered.get(), work.get(), PFFFT_FORWARD);
        pffft_zreorder(setup, spectrum.get(), output.get(), PFFFT_FORWARD);
        check_equal(output.get(), ordered.get(), count, 1.0);
        pffft_transform_ordered(setup, ordered.get(), output.get(), work.get(), PFFFT_BACKWARD);
        check_equal(output.get(), input.get(), count, 1.0 / n);
        std::memcpy(output.get(), input.get(), count * sizeof(float));
        pffft_transform(setup, output.get(), output.get(), work.get(), PFFFT_FORWARD);
        check_equal(output.get(), spectrum.get(), count, 1.0);
        pffft_transform(setup, output.get(), output.get(), work.get(), PFFFT_BACKWARD);
        check_equal(output.get(), input.get(), count, 1.0 / n);
        std::memcpy(output.get(), input.get(), count * sizeof(float));
        pffft_transform_ordered(setup, output.get(), output.get(), work.get(), PFFFT_FORWARD);
        check_equal(output.get(), ordered.get(), count, 1.0);
        pffft_transform_ordered(setup, output.get(), output.get(), work.get(), PFFFT_BACKWARD);
        check_equal(output.get(), input.get(), count, 1.0 / n);
        for (int i = 0; i < count; ++i) {
            filter[i] = spectrum[i] / n;
        }
        // Repeated unnormalized in-place transforms would overflow random input. Validate
        // with random data above, then time zeros so the measured loop needs no reset.
        if (op.find("inplace-") == 0) {
            std::memset(output.get(), 0, count * sizeof(float));
        }

#define MEASURE(...) measure(backend, argv[1], op, n, repeats, seconds, count, output.get(), [&]() { __VA_ARGS__; })
        if (op == "inplace-forward") {
            MEASURE(pffft_transform(setup, output.get(), output.get(), work.get(), PFFFT_FORWARD));
        } else if (op == "inplace-backward") {
            MEASURE(pffft_transform(setup, output.get(), output.get(), work.get(), PFFFT_BACKWARD));
        } else if (op == "inplace-ordered-forward") {
            MEASURE(pffft_transform_ordered(setup, output.get(), output.get(), work.get(), PFFFT_FORWARD));
        } else if (op == "inplace-ordered-backward") {
            MEASURE(pffft_transform_ordered(setup, output.get(), output.get(), work.get(), PFFFT_BACKWARD));
        } else if (op == "forward") {
            MEASURE(pffft_transform(setup, input.get(), output.get(), work.get(), PFFFT_FORWARD));
        } else if (op == "backward") {
            MEASURE(pffft_transform(setup, spectrum.get(), output.get(), work.get(), PFFFT_BACKWARD));
        } else if (op == "ordered-forward") {
            MEASURE(pffft_transform_ordered(setup, input.get(), output.get(), work.get(), PFFFT_FORWARD));
        } else if (op == "ordered-backward") {
            MEASURE(pffft_transform_ordered(setup, ordered.get(), output.get(), work.get(), PFFFT_BACKWARD));
        } else if (op == "reorder") {
            MEASURE(pffft_zreorder(setup, spectrum.get(), output.get(), PFFFT_FORWARD));
        } else if (op == "zconv") {
            MEASURE(pffft_zconvolve(setup, spectrum.get(), filter.get(), output.get()));
        } else if (op == "pipeline") {
            MEASURE(pffft_transform(setup, input.get(), spectrum.get(), work.get(), PFFFT_FORWARD);
                    pffft_zconvolve(setup, spectrum.get(), filter.get(), product.get());
                    pffft_transform(setup, product.get(), output.get(), work.get(), PFFFT_BACKWARD));
        } else {
            throw std::runtime_error("invalid arguments: unknown operation for the float driver");
        }
#undef MEASURE
        return 0;
    } catch (const std::exception& e) {
        (void)std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
