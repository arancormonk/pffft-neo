// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/* Modifications: Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com> */

/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.
*/

#ifndef PF_FLT_H
#define PF_FLT_H

#include <assert.h>
#include <string.h>
#include <stdint.h>

/*
 *  SIMD reference material:
 *
 * general SIMD introduction:
 * https://www.linuxjournal.com/content/introduction-gcc-compiler-intrinsics-vector-processing
 *
 * SSE 1:
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 *
 * ARM NEON:
 * https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
 *
 */

typedef float vsfscalar;

#include "pf_sse1_float.h"
#include "pf_neon_float.h"

#ifndef SIMD_SZ
#if !defined(PFFFT_SIMD_DISABLE)
#define PFFFT_SIMD_DISABLE /* fallback to scalar code */
#endif
#endif

#include "pf_scalar_float.h"

#ifndef VLD
/* vector load and store of p[i] in the shared pass codelets; the 8-lane
   AVX2 backend redefines them as explicit unaligned accesses */
#define VLD(p, i)    ((p)[i])
#define VST(p, i, v) ((p)[i] = (v))
#endif

/* shortcuts for complex multiplcations */
#define VCPLXMUL(ar, ai, br, bi)                                                                                       \
    {                                                                                                                  \
        v4sf tmp;                                                                                                      \
        tmp = VMUL(ar, bi);                                                                                            \
        (ar) = VMSUB(ai, bi, VMUL(ar, br));                                                                            \
        (ai) = VMADD(ai, br, tmp);                                                                                     \
    }
#define VCPLXMULCONJ(ar, ai, br, bi)                                                                                   \
    {                                                                                                                  \
        v4sf tmp;                                                                                                      \
        tmp = VMUL(ai, bi);                                                                                            \
        (ai) = VMSUB(ar, bi, VMUL(ai, br));                                                                            \
        (ar) = VMADD(ar, br, tmp);                                                                                     \
    }
/* (ar, ai) times (wr, -wi) for floats wr and wi: VCPLXMUL(ar, ai,
   LD_PS1(wr), LD_PS1(-wi)) without the negation. Like VCPLXMULCONJ it is
   the product with a conjugate, but it fuses the products that VCPLXMUL
   fuses and rounds the other two first, as VCPLXMUL does, so it has the
   bits of VCPLXMUL for the negated wi. It needs a*b - c, which every
   backend defines as VMSUBR(a, b, c), fused where its VMADD is
   (VMSUB(a, b, c) is c - a*b); a backend may define SVCPLXMUL_NEGI
   itself. wr and wi are lvalues, as LD_PS1 may take their address. */
#ifndef SVCPLXMUL_NEGI
#define SVCPLXMUL_NEGI(ar, ai, wr, wi)                                                                                 \
    {                                                                                                                  \
        const v4sf br_ = LD_PS1(wr), bi_ = LD_PS1(wi);                                                                 \
        v4sf tmp;                                                                                                      \
        tmp = VMUL(ar, bi_);                                                                                           \
        (ar) = VMADD(ai, bi_, VMUL(ar, br_));                                                                          \
        (ai) = VMSUBR(ai, br_, tmp);                                                                                   \
    }
#endif
/* Sums and a product with a sign: a + fsign*b, a - fsign*b and (ar, ai)
   times (wr, fsign*wi), for a float fsign that is +1 or -1, the direction
   of a complex transform, and floats wr and wi (lvalues). They have the
   bits of the multiplication by fsign and select the operation by it
   instead: where fsign is a constant, as it is in every compiled copy of
   a complex pass, one operation remains. */
#define VADD_FSIGN(a, b, fsign) ((fsign) > 0 ? VADD(a, b) : VSUB(a, b))
#define VSUB_FSIGN(a, b, fsign) ((fsign) > 0 ? VSUB(a, b) : VADD(a, b))
#define VCPLXMUL_FSIGN(ar, ai, wr, wi, fsign)                                                                          \
    {                                                                                                                  \
        if ((fsign) > 0) {                                                                                             \
            VCPLXMUL(ar, ai, LD_PS1(wr), LD_PS1(wi))                                                                   \
        } else {                                                                                                       \
            SVCPLXMUL_NEGI(ar, ai, wr, wi)                                                                             \
        }                                                                                                              \
    }
#ifndef SVMUL
/* multiply a scalar with a vector */
#define SVMUL(f, v) VMUL(LD_PS1(f), v)
#endif
#ifndef SVMADD
/* fused scalar-times-vector plus/minus a vector: c + f*v and c - f*v */
#define SVMADD(f, v, c) VMADD(LD_PS1(f), v, c)
#define SVMSUB(f, v, c) VMSUB(LD_PS1(f), v, c)
#endif

#endif /* PF_FLT_H */
