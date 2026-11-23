// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/* Modifications: Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com> */
/* Copyright (c) 2024  hayguen@web.de
   Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of PFFFT, nor the names of its
   sponsors or contributors may be used to endorse or promote products
   derived from this Software without specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
   THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
   DAMAGE.
*/

#ifndef PF_FMA_X86_H
#define PF_FMA_X86_H

/* SSE is optional on 32-bit x86. ARM64EC defines _M_X64 but targets Arm. */
#if !defined(_M_ARM64EC)                                                                                               \
    && (defined(__x86_64__) || defined(_M_X64) || defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
#define PFFFT_X86_HAVE_SSE 1
#endif
/*
  The float backend can use explicit FMA3 intrinsics on these targets.

  - gcc and clang define __FMA__ for -mfma, and for -march=haswell and
    later. Note that -mavx2 alone does NOT imply it.

  - MSVC defines no __FMA__ at all; it is absent from the predefined
    macro list. Its /arch:AVX2 (and /arch:AVX512, /arch:AVX10.*) does
    "enable use of Fused Multiply-Add (FMA) instructions", and every CPU
    with AVX2 has FMA3, so __AVX2__ is the equivalent gate there.

  - clang-cl also defines _MSC_VER, but unlike MSVC it rejects the
    intrinsic without the fma target feature, so it must go through
    __FMA__ instead of the MSVC branch.

  https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
  https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64
*/
#if !defined(PFFFT_DISABLE_FMA) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__) && defined(__AVX2__)))
#define PFFFT_X86_HAVE_FMA 1
#endif

#endif /* PF_FMA_X86_H */
