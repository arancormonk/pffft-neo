// SPDX-License-Identifier: GPL-2.0-or-later AND LicenseRef-PFFFT
/* Modifications: Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com> */
/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )
   Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB
   (http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
   of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

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


   PFFFT : a Pretty Fast FFT.

   This file is largerly based on the original FFTPACK implementation, modified in
   order to take advantage of SIMD instructions of modern CPUs.
*/

/* FFTPACK pass codelets, their drivers and the twiddle/factor setup.

   Included by pffft_priv_impl.h for the 4-lane backends (float, double) and
   again by pffft_wide_impl.h for the 8-lane AVX2 float backend, which renames
   every function defined here before the inclusion. The code only uses the
   vector macros (v4sf, VADD, VMUL, LD_PS1, VCPLXMUL, ...), so it works at any
   vector width. Deliberately no include guard: the file is meant to be
   included several times, and every inclusion is configured by macros that
   the including file defines before it and undefines after it, a pattern
   that CodeQL's cpp/missing-header-guard recognises as a template header.

   The eight complex codelets passf2/3/4/5 and passf2/3/4/5_il are
   always-inline bodies that take the direction of the transform, fsign,
   as a parameter. The functions passf2_ps, ..., passf5_il_ps hold a copy
   of their body for each direction (PF_PASS_DIRECTIONS below).

   The including file sets PF_INPLACE_CODELETS to 0 or 1, as it sets
   PF_FUSED_LAYOUT. With 0 the file compiles everything, the eight bodies
   as passf2_op_body, ..., passf5_il_op_body, with 'restrict'. With
   PFFFT_INPLACE_FIRST_PASS or PFFFT_DEPTH_FIRST and the fused layout,
   that inclusion includes the file once more, with PF_INPLACE_CODELETS 1,
   and sets it back to 0 afterwards. The inclusion with 1 compiles only
   the eight bodies, a second time, as passf2_body, ..., passf5_il_body
   without 'restrict', for the in-place first-pass wrappers (passf2_ip_ps,
   ..., passf5_il_ip_ps); only the function header changes, through
   PF_CODELET_NAME and PF_CODELET_RESTRICT. Only the eight bodies (with
   their local macros), the definitions of those two macros and the
   #ifdef PF_FUSED_LAYOUT around the four _il bodies are outside
   #if !PF_INPLACE_CODELETS. */

#ifndef PF_INPLACE_CODELETS
#error "pffft_passes_impl.h: the including file has to define PF_INPLACE_CODELETS to 0 or 1"
#endif

#if !PF_INPLACE_CODELETS
#include "pffft_pass_config.h" /* PFFFT_TERMINAL_RADIX8, PFFFT_INPLACE_FIRST_PASS */
#endif

/* the header of the eight codelet bodies compiled twice: for the regular
   codelets, or for the in-place wrappers */
#undef PF_CODELET_NAME
#undef PF_CODELET_RESTRICT
#if PF_INPLACE_CODELETS
#define PF_CODELET_NAME(name) name##_body
#define PF_CODELET_RESTRICT
#else
#define PF_CODELET_NAME(name) name##_op_body
#define PF_CODELET_RESTRICT   RESTRICT

/*
  The direction of a complex pass: fsign is -1 for the forward transform
  and +1 for the backward one. All a pass does with it is multiply by it:
  the imaginary parts of its twiddles, and differences or constants of its
  butterfly. In its loops a body does that with VADD_FSIGN, VSUB_FSIGN and
  VCPLXMUL_FSIGN (simd/pf_float.h), which select the operation by the sign
  and give the bits of the multiplication. Every function that holds a
  body calls it once for each sign, with fsign a constant, so that the
  compiler keeps the operations of that direction alone: the function has
  a forward and a backward copy of the pass, and neither multiplies by
  fsign in a loop.

  PF_PASS_DIRECTIONS(fsign, body, ...) is that pair of calls,
  body(..., +1) and body(..., -1), for the fsign of the function.
*/
#define PF_PASS_DIRECTIONS(fsign, body, ...)                                                                           \
    {                                                                                                                  \
        if ((fsign) > 0) {                                                                                             \
            body(__VA_ARGS__, +1.f);                                                                                   \
        } else {                                                                                                       \
            body(__VA_ARGS__, -1.f);                                                                                   \
        }                                                                                                              \
    }
#endif

/*
  passf2 and passb2 has been merged here, fsign = -1 for passf2, +1 for passb2
*/
static ALWAYS_INLINE(void) PF_CODELET_NAME(passf2)(int ido, int l1, const v4sf* PF_CODELET_RESTRICT cc,
                                                   v4sf* PF_CODELET_RESTRICT ch, const float* wa1, float fsign) {
    int k, i;
    int l1ido = l1 * ido;
    if (ido <= 2) {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 2 * ido) {
            VST(ch, 0, VADD(VLD(cc, 0), VLD(cc, ido + 0)));
            VST(ch, l1ido, VSUB(VLD(cc, 0), VLD(cc, ido + 0)));
            VST(ch, 1, VADD(VLD(cc, 1), VLD(cc, ido + 1)));
            VST(ch, l1ido + 1, VSUB(VLD(cc, 1), VLD(cc, ido + 1)));
        }
    } else {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 2 * ido) {
            for (i = 0; i < ido - 1; i += 2) {
                v4sf tr2 = VSUB(VLD(cc, i + 0), VLD(cc, i + ido + 0));
                v4sf ti2 = VSUB(VLD(cc, i + 1), VLD(cc, i + ido + 1));
                const float wr = wa1[i], wi = wa1[i + 1];
                VST(ch, i, VADD(VLD(cc, i + 0), VLD(cc, i + ido + 0)));
                VST(ch, i + 1, VADD(VLD(cc, i + 1), VLD(cc, i + ido + 1)));
                VCPLXMUL_FSIGN(tr2, ti2, wr, wi, fsign);
                VST(ch, i + l1ido, tr2);
                VST(ch, i + l1ido + 1, ti2);
            }
        }
    }
}

/*
  passf3 and passb3 has been merged here, fsign = -1 for passf3, +1 for passb3

  The radix-3/5 twiddle constants are written as full-precision double literals
  without an 'f' suffix, for when this file is compiled with '#define float double'.
  In the float build, the values round to the same floats as the old suffixed literals.
*/
static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf3)(int ido, int l1, const v4sf* PF_CODELET_RESTRICT cc, v4sf* PF_CODELET_RESTRICT ch,
                            const float* wa1, const float* wa2, float fsign) {
    static const float taur = (float)-0.5;              /* cos(2*pi/3)   */
    float taui = (float)0.86602540378443864676 * fsign; /* sin(2*pi/3) = sqrt(3)/2 */
    int i, k;
    v4sf tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
    int l1ido = l1 * ido;
    float wr1, wi1, wr2, wi2;
    assert(ido >= 2); /* ido == 2 (one complex element) is reached when SIMD_SZ == 1 */
    for (k = 0; k < l1ido; k += ido, cc += 3 * ido, ch += ido) {
        for (i = 0; i < ido - 1; i += 2) {
            tr2 = VADD(VLD(cc, i + ido), VLD(cc, i + 2 * ido));
            cr2 = SVMADD(taur, tr2, VLD(cc, i));
            VST(ch, i, VADD(VLD(cc, i), tr2));
            ti2 = VADD(VLD(cc, i + ido + 1), VLD(cc, i + 2 * ido + 1));
            ci2 = SVMADD(taur, ti2, VLD(cc, i + 1));
            VST(ch, i + 1, VADD(VLD(cc, i + 1), ti2));
            cr3 = SVMUL(taui, VSUB(VLD(cc, i + ido), VLD(cc, i + 2 * ido)));
            ci3 = SVMUL(taui, VSUB(VLD(cc, i + ido + 1), VLD(cc, i + 2 * ido + 1)));
            dr2 = VSUB(cr2, ci3);
            dr3 = VADD(cr2, ci3);
            di2 = VADD(ci2, cr3);
            di3 = VSUB(ci2, cr3);
            wr1 = wa1[i];
            wi1 = wa1[i + 1];
            wr2 = wa2[i];
            wi2 = wa2[i + 1];
            VCPLXMUL_FSIGN(dr2, di2, wr1, wi1, fsign);
            VST(ch, i + l1ido, dr2);
            VST(ch, i + l1ido + 1, di2);
            VCPLXMUL_FSIGN(dr3, di3, wr2, wi2, fsign);
            VST(ch, i + 2 * l1ido, dr3);
            VST(ch, i + 2 * l1ido + 1, di3);
        }
    }
} /* passf3 */

static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf4)(int ido, int l1, const v4sf* PF_CODELET_RESTRICT cc, v4sf* PF_CODELET_RESTRICT ch,
                            const float* wa1, const float* wa2, const float* wa3, float fsign) {
    /* fsign == -1 for forward transform and +1 for backward transform; tr4
       and ti4 are the differences that FFTPACK multiplies by it */

    int i, k;
    v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    int l1ido = l1 * ido;
    if (ido == 2) {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido) {
            tr1 = VSUB(VLD(cc, 0), VLD(cc, 2 * ido + 0));
            tr2 = VADD(VLD(cc, 0), VLD(cc, 2 * ido + 0));
            ti1 = VSUB(VLD(cc, 1), VLD(cc, 2 * ido + 1));
            ti2 = VADD(VLD(cc, 1), VLD(cc, 2 * ido + 1));
            ti4 = VSUB(VLD(cc, 1 * ido + 0), VLD(cc, 3 * ido + 0));
            tr4 = VSUB(VLD(cc, 3 * ido + 1), VLD(cc, 1 * ido + 1));
            tr3 = VADD(VLD(cc, ido + 0), VLD(cc, 3 * ido + 0));
            ti3 = VADD(VLD(cc, ido + 1), VLD(cc, 3 * ido + 1));

            VST(ch, 0 * l1ido + 0, VADD(tr2, tr3));
            VST(ch, 0 * l1ido + 1, VADD(ti2, ti3));
            VST(ch, 1 * l1ido + 0, VADD_FSIGN(tr1, tr4, fsign));
            VST(ch, 1 * l1ido + 1, VADD_FSIGN(ti1, ti4, fsign));
            VST(ch, 2 * l1ido + 0, VSUB(tr2, tr3));
            VST(ch, 2 * l1ido + 1, VSUB(ti2, ti3));
            VST(ch, 3 * l1ido + 0, VSUB_FSIGN(tr1, tr4, fsign));
            VST(ch, 3 * l1ido + 1, VSUB_FSIGN(ti1, ti4, fsign));
        }
    } else {
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 4 * ido) {
            for (i = 0; i < ido - 1; i += 2) {
                float wr1, wi1, wr2, wi2, wr3, wi3;
                tr1 = VSUB(VLD(cc, i + 0), VLD(cc, i + 2 * ido + 0));
                tr2 = VADD(VLD(cc, i + 0), VLD(cc, i + 2 * ido + 0));
                ti1 = VSUB(VLD(cc, i + 1), VLD(cc, i + 2 * ido + 1));
                ti2 = VADD(VLD(cc, i + 1), VLD(cc, i + 2 * ido + 1));
                tr4 = VSUB(VLD(cc, i + 3 * ido + 1), VLD(cc, i + 1 * ido + 1));
                ti4 = VSUB(VLD(cc, i + 1 * ido + 0), VLD(cc, i + 3 * ido + 0));
                tr3 = VADD(VLD(cc, i + ido + 0), VLD(cc, i + 3 * ido + 0));
                ti3 = VADD(VLD(cc, i + ido + 1), VLD(cc, i + 3 * ido + 1));

                VST(ch, i, VADD(tr2, tr3));
                cr3 = VSUB(tr2, tr3);
                VST(ch, i + 1, VADD(ti2, ti3));
                ci3 = VSUB(ti2, ti3);

                cr2 = VADD_FSIGN(tr1, tr4, fsign);
                cr4 = VSUB_FSIGN(tr1, tr4, fsign);
                ci2 = VADD_FSIGN(ti1, ti4, fsign);
                ci4 = VSUB_FSIGN(ti1, ti4, fsign);
                wr1 = wa1[i];
                wi1 = wa1[i + 1];
                VCPLXMUL_FSIGN(cr2, ci2, wr1, wi1, fsign);
                wr2 = wa2[i];
                wi2 = wa2[i + 1];
                VST(ch, i + l1ido, cr2);
                VST(ch, i + l1ido + 1, ci2);

                VCPLXMUL_FSIGN(cr3, ci3, wr2, wi2, fsign);
                wr3 = wa3[i];
                wi3 = wa3[i + 1];
                VST(ch, i + 2 * l1ido, cr3);
                VST(ch, i + 2 * l1ido + 1, ci3);

                VCPLXMUL_FSIGN(cr4, ci4, wr3, wi3, fsign);
                VST(ch, i + 3 * l1ido, cr4);
                VST(ch, i + 3 * l1ido + 1, ci4);
            }
        }
    }
} /* passf4 */

/*
  passf5 and passb5 has been merged here, fsign = -1 for passf5, +1 for passb5
*/
static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf5)(int ido, int l1, const v4sf* PF_CODELET_RESTRICT cc, v4sf* PF_CODELET_RESTRICT ch,
                            const float* wa1, const float* wa2, const float* wa3, const float* wa4, float fsign) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    const float ti11 = (float)0.95105651629515357212 * fsign; /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    const float ti12 = (float)0.58778525229247312917 * fsign; /* sin(4*pi/5) */

    /* Local variables */
    int i, k;
    v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3,
        tr4, tr5;

    float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

#define cc_idx(a_1, a_2) (((a_2) - 1) * ido + (a_1) + 1)
#define ch_idx(a_1, a_3) (((a_3) - 1) * l1 * ido + (a_1) + 1)
/* leg 1, read again after the first two stores; at l1 == 1 these store to
   the positions of leg 1, so in place it is loaded before them */
#if PF_INPLACE_CODELETS
#define cc_leg1r c1r
#define cc_leg1i c1i
#else
#define cc_leg1r VLD(cc, cc_idx(i - 1, 1))
#define cc_leg1i VLD(cc, cc_idx(i, 1))
#endif

    assert(ido >= 2); /* ido == 2 (one complex element) is reached when SIMD_SZ == 1 */
    for (k = 0; k < l1; ++k, cc += 5 * ido, ch += ido) {
        for (i = 0; i < ido - 1; i += 2) {
            ti5 = VSUB(VLD(cc, cc_idx(i, 2)), VLD(cc, cc_idx(i, 5)));
            ti2 = VADD(VLD(cc, cc_idx(i, 2)), VLD(cc, cc_idx(i, 5)));
            ti4 = VSUB(VLD(cc, cc_idx(i, 3)), VLD(cc, cc_idx(i, 4)));
            ti3 = VADD(VLD(cc, cc_idx(i, 3)), VLD(cc, cc_idx(i, 4)));
            tr5 = VSUB(VLD(cc, cc_idx(i - 1, 2)), VLD(cc, cc_idx(i - 1, 5)));
            tr2 = VADD(VLD(cc, cc_idx(i - 1, 2)), VLD(cc, cc_idx(i - 1, 5)));
            tr4 = VSUB(VLD(cc, cc_idx(i - 1, 3)), VLD(cc, cc_idx(i - 1, 4)));
            tr3 = VADD(VLD(cc, cc_idx(i - 1, 3)), VLD(cc, cc_idx(i - 1, 4)));
#if PF_INPLACE_CODELETS
            const v4sf c1r = VLD(cc, cc_idx(i - 1, 1)), c1i = VLD(cc, cc_idx(i, 1));
#endif
            VST(ch, ch_idx(i - 1, 1), VADD(VLD(cc, cc_idx(i - 1, 1)), VADD(tr2, tr3)));
            VST(ch, ch_idx(i, 1), VADD(VLD(cc, cc_idx(i, 1)), VADD(ti2, ti3)));
            cr2 = VADD(cc_leg1r, SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
            ci2 = VADD(cc_leg1i, SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
            cr3 = VADD(cc_leg1r, SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
            ci3 = VADD(cc_leg1i, SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
            cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
            ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
            cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
            ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
            dr3 = VSUB(cr3, ci4);
            dr4 = VADD(cr3, ci4);
            di3 = VADD(ci3, cr4);
            di4 = VSUB(ci3, cr4);
            dr5 = VADD(cr2, ci5);
            dr2 = VSUB(cr2, ci5);
            di5 = VSUB(ci2, cr5);
            di2 = VADD(ci2, cr5);
            wr1 = wa1[i];
            wi1 = wa1[i + 1];
            wr2 = wa2[i];
            wi2 = wa2[i + 1];
            wr3 = wa3[i];
            wi3 = wa3[i + 1];
            wr4 = wa4[i];
            wi4 = wa4[i + 1];
            VCPLXMUL_FSIGN(dr2, di2, wr1, wi1, fsign);
            VST(ch, ch_idx(i - 1, 2), dr2);
            VST(ch, ch_idx(i, 2), di2);
            VCPLXMUL_FSIGN(dr3, di3, wr2, wi2, fsign);
            VST(ch, ch_idx(i - 1, 3), dr3);
            VST(ch, ch_idx(i, 3), di3);
            VCPLXMUL_FSIGN(dr4, di4, wr3, wi3, fsign);
            VST(ch, ch_idx(i - 1, 4), dr4);
            VST(ch, ch_idx(i, 4), di4);
            VCPLXMUL_FSIGN(dr5, di5, wr4, wi4, fsign);
            VST(ch, ch_idx(i - 1, 5), dr5);
            VST(ch, ch_idx(i, 5), di5);
        }
    }
#undef cc_leg1i
#undef cc_leg1r
#undef ch_idx
#undef cc_idx
}

#if !PF_INPLACE_CODELETS
/* the four codelets as functions, for either direction */
static NEVER_INLINE(void)
    passf2_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf2_op_body, ido, l1, cc, ch, wa1);
}

static NEVER_INLINE(void) passf3_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                    const float* wa2, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf3_op_body, ido, l1, cc, ch, wa1, wa2);
}

static NEVER_INLINE(void) passf4_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                    const float* wa2, const float* wa3, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf4_op_body, ido, l1, cc, ch, wa1, wa2, wa3);
}

static NEVER_INLINE(void) passf5_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                    const float* wa2, const float* wa3, const float* wa4, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf5_op_body, ido, l1, cc, ch, wa1, wa2, wa3, wa4);
}

#if PFFFT_TERMINAL_RADIX8
/*
  passf8 and passb8 merged here, fsign = -1 for passf8, +1 for passb8.

  The radix-8 pass of terminal_merge_plan: the last pass of a complex
  transform, with one complex element per leg (ido == 2) and therefore no
  twiddles, for any l1. Where the plan has no 3 (else see passf6_ps), it
  replaces the plan's radix-2 pass and its last radix-4 pass: the radix-2
  pass is dropped and the last radix-4 pass becomes this one, so every
  pass in between runs with other l1 and ido; the transform has one pass
  over the data fewer.

  The radix-8 DFT is evaluated half by half: the even outputs are the DFT-4
  of (x_j + x_{j+4}), the odd ones the DFT-4 of (x_j - x_{j+4}) w8^j. The
  inputs are loaded again for the odd half rather than kept in registers,
  so that at most about twelve vector values are live (x86-64 has sixteen
  vector registers); cc and ch are deliberately not 'restrict', which makes
  the compiler reload them after the first half's stores. The 1/sqrt(2) of
  w8 and w8^3 is folded into the last butterfly as a multiply-add.

  The backward transform is the forward one with real and imaginary parts
  exchanged on input and output: R, I are the vector offsets of the parts
  the arithmetic treats as real and imaginary (0,1 forward; 1,0 backward),
  and ST(m, a, b) stores output m, a being the part read at offset R.
  k8 is the vector 1/sqrt(2) in the caller.
*/
#define PF_PASS8(cc, R, I, ST)                                                                                         \
    {                                                                                                                  \
        v4sf s0r = VADD(VLD(cc, 0 + (R)), VLD(cc, 8 + (R))), s0i = VADD(VLD(cc, 0 + (I)), VLD(cc, 8 + (I)));           \
        v4sf s1r = VADD(VLD(cc, 2 + (R)), VLD(cc, 10 + (R))), s1i = VADD(VLD(cc, 2 + (I)), VLD(cc, 10 + (I)));         \
        v4sf s2r = VADD(VLD(cc, 4 + (R)), VLD(cc, 12 + (R))), s2i = VADD(VLD(cc, 4 + (I)), VLD(cc, 12 + (I)));         \
        v4sf s3r = VADD(VLD(cc, 6 + (R)), VLD(cc, 14 + (R))), s3i = VADD(VLD(cc, 6 + (I)), VLD(cc, 14 + (I)));         \
        v4sf t1r = VSUB(s0r, s2r), t1i = VSUB(s0i, s2i);                                                               \
        v4sf t2r = VADD(s0r, s2r), t2i = VADD(s0i, s2i);                                                               \
        v4sf t3r = VADD(s1r, s3r), t3i = VADD(s1i, s3i);                                                               \
        v4sf t4r = VSUB(s1i, s3i), t4i = VSUB(s3r, s1r);                                                               \
        ST(0, VADD(t2r, t3r), VADD(t2i, t3i))                                                                          \
        ST(2, VADD(t1r, t4r), VADD(t1i, t4i))                                                                          \
        ST(4, VSUB(t2r, t3r), VSUB(t2i, t3i))                                                                          \
        ST(6, VSUB(t1r, t4r), VSUB(t1i, t4i))                                                                          \
    }                                                                                                                  \
    {                                                                                                                  \
        v4sf d0r = VSUB(VLD(cc, 0 + (R)), VLD(cc, 8 + (R))), d0i = VSUB(VLD(cc, 0 + (I)), VLD(cc, 8 + (I)));           \
        v4sf d1r = VSUB(VLD(cc, 2 + (R)), VLD(cc, 10 + (R))), d1i = VSUB(VLD(cc, 2 + (I)), VLD(cc, 10 + (I)));         \
        v4sf d2r = VSUB(VLD(cc, 4 + (R)), VLD(cc, 12 + (R))), d2i = VSUB(VLD(cc, 4 + (I)), VLD(cc, 12 + (I)));         \
        v4sf d3r = VSUB(VLD(cc, 6 + (R)), VLD(cc, 14 + (R))), d3i = VSUB(VLD(cc, 6 + (I)), VLD(cc, 14 + (I)));         \
        v4sf p1r = VADD(d1r, d1i), p1i = VSUB(d1i, d1r);                                                               \
        v4sf s3_ = VADD(d3r, d3i), p3r = VSUB(d3i, d3r);                                                               \
        v4sf t1r = VSUB(d0r, d2i), t1i = VADD(d0i, d2r);                                                               \
        v4sf t2r = VADD(d0r, d2i), t2i = VSUB(d0i, d2r);                                                               \
        v4sf q3r = VADD(p1r, p3r), q3i = VSUB(p1i, s3_);                                                               \
        v4sf er_ = VSUB(p1r, p3r), ei_ = VADD(p1i, s3_);                                                               \
        ST(1, VMADD(k8, q3r, t2r), VMADD(k8, q3i, t2i))                                                                \
        ST(3, VMADD(k8, ei_, t1r), VMSUB(k8, er_, t1i))                                                                \
        ST(5, VMSUB(k8, q3r, t2r), VMSUB(k8, q3i, t2i))                                                                \
        ST(7, VMSUB(k8, ei_, t1r), VMADD(k8, er_, t1i))                                                                \
    }

static NEVER_INLINE(void) passf8_ps(int ido, int l1, const v4sf* cc, v4sf* ch, float fsign) {
    const float c8 = (float)M_SQRT1_2; /* cos(pi/4) = sin(pi/4) = 1/sqrt(2) */
    const v4sf k8 = LD_PS1(c8);        /* LD_PS1 may take its argument's address */
    const int l1ido = l1 * ido;
    int k;
    assert(ido == 2);
    if (fsign < 0) {
#define PF_ST8(m, a, b)                                                                                                \
    {                                                                                                                  \
        VST(ch, (m) * l1ido, a);                                                                                       \
        VST(ch, (m) * l1ido + 1, b);                                                                                   \
    }
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 8 * ido) {
            PF_PASS8(cc, 0, 1, PF_ST8)
        }
#undef PF_ST8
    } else {
#define PF_ST8(m, a, b)                                                                                                \
    {                                                                                                                  \
        VST(ch, (m) * l1ido + 1, a);                                                                                   \
        VST(ch, (m) * l1ido, b);                                                                                       \
    }
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 8 * ido) {
            PF_PASS8(cc, 1, 0, PF_ST8)
        }
#undef PF_ST8
    }
} /* passf8 */
#else  /* PFFFT_TERMINAL_RADIX8 */
/*
  passf8 and passb8 merged here, fsign = -1 for passf8, +1 for passb8

  Radix-8 DFT via two radix-4 DFTs:
    even outputs from DFT-4 of (x0+x4, x1+x5, x2+x6, x3+x7)
    odd  outputs from DFT-4 of (x0-x4, x1-x5, x2-x6, x3-x7)*W8^k
*/
/*
  cffti1_ps only ever selects factor 8 when n == 8 (the Ncvec == 8,
  N == 32 case); at the sole dispatch site below this forces ido == 2
  and l1 == 1. This codelet is therefore specialized to that single
  reachable shape instead of the general ido/l1 loop: the assert
  enforces the invariant at the point of use.
*/
static ALWAYS_INLINE(void) passf8_op_body(int ido, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, float fsign) {
    static const float K8 = (float)M_SQRT1_2; /* cos(pi/4) = sin(pi/4) = 1/sqrt(2) */
    const float nK8 = -K8;

    /* Load 8 inputs: cc[leg*ido + 0] = real, cc[leg*ido + 1] = imag */
    v4sf x0r = VLD(cc, 0 * ido + 0), x0i = VLD(cc, 0 * ido + 1);
    v4sf x1r = VLD(cc, 1 * ido + 0), x1i = VLD(cc, 1 * ido + 1);
    v4sf x2r = VLD(cc, 2 * ido + 0), x2i = VLD(cc, 2 * ido + 1);
    v4sf x3r = VLD(cc, 3 * ido + 0), x3i = VLD(cc, 3 * ido + 1);
    v4sf x4r = VLD(cc, 4 * ido + 0), x4i = VLD(cc, 4 * ido + 1);
    v4sf x5r = VLD(cc, 5 * ido + 0), x5i = VLD(cc, 5 * ido + 1);
    v4sf x6r = VLD(cc, 6 * ido + 0), x6i = VLD(cc, 6 * ido + 1);
    v4sf x7r = VLD(cc, 7 * ido + 0), x7i = VLD(cc, 7 * ido + 1);

    /* Step 1: sums and diffs at stride 4 */
    v4sf s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
    v4sf s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
    v4sf s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
    v4sf s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
    v4sf d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
    v4sf d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
    v4sf d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
    v4sf d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);

    /* Step 2: twiddle diffs by W8^k */
    /* d0 *= W8^0 = 1 (no-op) */
    /* d1 *= W8^1 = (K8, fsign*K8) */
    VCPLXMUL_FSIGN(d1r, d1i, K8, K8, fsign);
    /* d2 *= W8^2 = (0, fsign), which makes it (-fsign*d2i, fsign*d2r): the
       sums and differences of step 4 take d2i and d2r with those signs */
    /* d3 *= W8^3 = (-K8, fsign*K8) */
    VCPLXMUL_FSIGN(d3r, d3i, nK8, K8, fsign);

    /* Step 3: DFT-4 on sums -> even outputs (0,2,4,6); t4r and t4i are the
       differences that are multiplied by fsign */
    {
        v4sf t1r = VSUB(s0r, s2r), t1i = VSUB(s0i, s2i);
        v4sf t2r = VADD(s0r, s2r), t2i = VADD(s0i, s2i);
        v4sf t4i = VSUB(s1r, s3r);
        v4sf t4r = VSUB(s3i, s1i);
        v4sf t3r = VADD(s1r, s3r), t3i = VADD(s1i, s3i);

        VST(ch, 0 * ido + 0, VADD(t2r, t3r));
        VST(ch, 0 * ido + 1, VADD(t2i, t3i));
        VST(ch, 2 * ido + 0, VADD_FSIGN(t1r, t4r, fsign));
        VST(ch, 2 * ido + 1, VADD_FSIGN(t1i, t4i, fsign));
        VST(ch, 4 * ido + 0, VSUB(t2r, t3r));
        VST(ch, 4 * ido + 1, VSUB(t2i, t3i));
        VST(ch, 6 * ido + 0, VSUB_FSIGN(t1r, t4r, fsign));
        VST(ch, 6 * ido + 1, VSUB_FSIGN(t1i, t4i, fsign));
    }

    /* Step 4: DFT-4 on diffs -> odd outputs (1,3,5,7) */
    {
        v4sf t1r = VADD_FSIGN(d0r, d2i, fsign), t1i = VSUB_FSIGN(d0i, d2r, fsign);
        v4sf t2r = VSUB_FSIGN(d0r, d2i, fsign), t2i = VADD_FSIGN(d0i, d2r, fsign);
        v4sf t4i = VSUB(d1r, d3r);
        v4sf t4r = VSUB(d3i, d1i);
        v4sf t3r = VADD(d1r, d3r), t3i = VADD(d1i, d3i);

        VST(ch, 1 * ido + 0, VADD(t2r, t3r));
        VST(ch, 1 * ido + 1, VADD(t2i, t3i));
        VST(ch, 3 * ido + 0, VADD_FSIGN(t1r, t4r, fsign));
        VST(ch, 3 * ido + 1, VADD_FSIGN(t1i, t4i, fsign));
        VST(ch, 5 * ido + 0, VSUB(t2r, t3r));
        VST(ch, 5 * ido + 1, VSUB(t2i, t3i));
        VST(ch, 7 * ido + 0, VSUB_FSIGN(t1r, t4r, fsign));
        VST(ch, 7 * ido + 1, VSUB_FSIGN(t1i, t4i, fsign));
    }
}

static NEVER_INLINE(void) passf8_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, float fsign) {
    assert(ido == 2 && l1 == 1);
    (void)l1; /* otherwise unused when assertions are disabled */
    PF_PASS_DIRECTIONS(fsign, passf8_op_body, ido, cc, ch);
} /* passf8 */
#endif /* PFFFT_TERMINAL_RADIX8 */

#if PFFFT_TERMINAL_RADIX8
/*
  passf6 and passb6 merged here, fsign = -1 for passf6, +1 for passb6.

  The radix-6 pass of terminal_merge_plan: the last pass of a complex
  transform, with one complex element per leg (ido == 2) and no twiddles,
  for any l1, in place of the radix-2 pass and a radix-3 one. 2 and 3 are
  coprime, so the 6-point DFT needs no twiddles inside either
  (Good-Thomas): with the input index n = 3 n1 + 2 n2 and the output index
  m = m1 mod 2, m = m2 mod 3, w6^(n m) = (-1)^(n1 m1) w3^(n2 m2). A DFT-2
  of each pair (x0, x3), (x2, x5), (x4, x1), then a DFT-3 of the sums
  gives y0, y4, y2 and one of the differences y3, y1, y5. Per k: 32
  adds and subtracts, 4 multiply-adds and 4 multiplies; passf8_ps takes
  44 adds and subtracts and 8 multiply-adds for 8 legs.

  The backward transform is the forward one with real and imaginary parts
  exchanged on input and output, as in PF_PASS8: R, I are the offsets of
  the parts the arithmetic treats as real and imaginary, ST(m, a, b)
  stores output m, a being the part read at offset R. taur, taui are the
  vectors cos(2 pi/3), sin(2 pi/3) in the caller. As for passf8_ps, cc
  and ch are not 'restrict', so that the second half loads its inputs
  again rather than keeping them live.
*/
#define PF_PASS6(cc, R, I, ST)                                                                                         \
    {                                                                                                                  \
        v4sf s0r = VADD(VLD(cc, 0 + (R)), VLD(cc, 6 + (R))), s0i = VADD(VLD(cc, 0 + (I)), VLD(cc, 6 + (I)));           \
        v4sf s1r = VADD(VLD(cc, 4 + (R)), VLD(cc, 10 + (R))), s1i = VADD(VLD(cc, 4 + (I)), VLD(cc, 10 + (I)));         \
        v4sf s2r = VADD(VLD(cc, 8 + (R)), VLD(cc, 2 + (R))), s2i = VADD(VLD(cc, 8 + (I)), VLD(cc, 2 + (I)));           \
        v4sf tr = VADD(s1r, s2r), ti = VADD(s1i, s2i);                                                                 \
        v4sf cr = VMADD(taur, tr, s0r), ci = VMADD(taur, ti, s0i);                                                     \
        v4sf ur = VMUL(taui, VSUB(s1r, s2r)), ui = VMUL(taui, VSUB(s1i, s2i));                                         \
        ST(0, VADD(s0r, tr), VADD(s0i, ti))                                                                            \
        ST(4, VADD(cr, ui), VSUB(ci, ur))                                                                              \
        ST(2, VSUB(cr, ui), VADD(ci, ur))                                                                              \
    }                                                                                                                  \
    {                                                                                                                  \
        v4sf d0r = VSUB(VLD(cc, 0 + (R)), VLD(cc, 6 + (R))), d0i = VSUB(VLD(cc, 0 + (I)), VLD(cc, 6 + (I)));           \
        v4sf d1r = VSUB(VLD(cc, 4 + (R)), VLD(cc, 10 + (R))), d1i = VSUB(VLD(cc, 4 + (I)), VLD(cc, 10 + (I)));         \
        v4sf d2r = VSUB(VLD(cc, 8 + (R)), VLD(cc, 2 + (R))), d2i = VSUB(VLD(cc, 8 + (I)), VLD(cc, 2 + (I)));           \
        v4sf tr = VADD(d1r, d2r), ti = VADD(d1i, d2i);                                                                 \
        v4sf cr = VMADD(taur, tr, d0r), ci = VMADD(taur, ti, d0i);                                                     \
        v4sf ur = VMUL(taui, VSUB(d1r, d2r)), ui = VMUL(taui, VSUB(d1i, d2i));                                         \
        ST(3, VADD(d0r, tr), VADD(d0i, ti))                                                                            \
        ST(1, VADD(cr, ui), VSUB(ci, ur))                                                                              \
        ST(5, VSUB(cr, ui), VADD(ci, ur))                                                                              \
    }

static NEVER_INLINE(void) passf6_ps(int ido, int l1, const v4sf* cc, v4sf* ch, float fsign) {
    const float ctaur = -0.5f, ctaui = (float)0.86602540378443864676; /* cos, sin(2*pi/3) */
    const v4sf taur = LD_PS1(ctaur), taui = LD_PS1(ctaui);            /* named: see passf8_ps */
    const int l1ido = l1 * ido;
    int k;
    assert(ido == 2);
    if (fsign < 0) {
#define PF_ST6(m, a, b)                                                                                                \
    {                                                                                                                  \
        VST(ch, (m) * l1ido, a);                                                                                       \
        VST(ch, (m) * l1ido + 1, b);                                                                                   \
    }
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 6 * ido) {
            PF_PASS6(cc, 0, 1, PF_ST6)
        }
#undef PF_ST6
    } else {
#define PF_ST6(m, a, b)                                                                                                \
    {                                                                                                                  \
        VST(ch, (m) * l1ido + 1, a);                                                                                   \
        VST(ch, (m) * l1ido, b);                                                                                       \
    }
        for (k = 0; k < l1ido; k += ido, ch += ido, cc += 6 * ido) {
            PF_PASS6(cc, 1, 0, PF_ST6)
        }
#undef PF_ST6
    }
} /* passf6 */
#endif /* PFFFT_TERMINAL_RADIX8 */

static NEVER_INLINE(void) radf2_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1) {
    static const float minus_one = -1.f;
    int i, k, l1ido = l1 * ido;
    for (k = 0; k < l1ido; k += ido) {
        v4sf a = VLD(cc, k), b = VLD(cc, k + l1ido);
        VST(ch, 2 * k, VADD(a, b));
        VST(ch, 2 * (k + ido) - 1, VSUB(a, b));
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (k = 0; k < l1ido; k += ido) {
            for (i = 2; i < ido; i += 2) {
                v4sf tr2 = VLD(cc, i - 1 + k + l1ido), ti2 = VLD(cc, i + k + l1ido);
                v4sf br = VLD(cc, i - 1 + k), bi = VLD(cc, i + k);
                VCPLXMULCONJ(tr2, ti2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
                VST(ch, i + 2 * k, VADD(bi, ti2));
                VST(ch, 2 * (k + ido) - i, VSUB(ti2, bi));
                VST(ch, i - 1 + 2 * k, VADD(br, tr2));
                VST(ch, 2 * (k + ido) - i - 1, VSUB(br, tr2));
            }
        }
        if (ido % 2 != 0) {
            return;
        }
    }
    for (k = 0; k < l1ido; k += ido) {
        VST(ch, 2 * k + ido, SVMUL(minus_one, VLD(cc, ido - 1 + k + l1ido)));
        VST(ch, 2 * k + ido - 1, VLD(cc, k + ido - 1));
    }
} /* radf2 */

static NEVER_INLINE(void) radb2_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1) {
    static const float minus_two = -2;
    int i, k, l1ido = l1 * ido;
    v4sf a, b, c, d, tr2, ti2;
    for (k = 0; k < l1ido; k += ido) {
        a = VLD(cc, 2 * k);
        b = VLD(cc, 2 * (k + ido) - 1);
        VST(ch, k, VADD(a, b));
        VST(ch, k + l1ido, VSUB(a, b));
    }
    if (ido < 2) {
        return;
    }
    if (ido != 2) {
        for (k = 0; k < l1ido; k += ido) {
            for (i = 2; i < ido; i += 2) {
                a = VLD(cc, i - 1 + 2 * k);
                b = VLD(cc, 2 * (k + ido) - i - 1);
                c = VLD(cc, i + 0 + 2 * k);
                d = VLD(cc, 2 * (k + ido) - i + 0);
                VST(ch, i - 1 + k, VADD(a, b));
                tr2 = VSUB(a, b);
                VST(ch, i + 0 + k, VSUB(c, d));
                ti2 = VADD(c, d);
                VCPLXMUL(tr2, ti2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
                VST(ch, i - 1 + k + l1ido, tr2);
                VST(ch, i + 0 + k + l1ido, ti2);
            }
        }
        if (ido % 2 != 0) {
            return;
        }
    }
    for (k = 0; k < l1ido; k += ido) {
        a = VLD(cc, 2 * k + ido - 1);
        b = VLD(cc, 2 * k + ido);
        VST(ch, k + ido - 1, VADD(a, a));
        VST(ch, k + ido - 1 + l1ido, SVMUL(minus_two, b));
    }
} /* radb2 */

/*
  The odd-radix real passes, radf3/radb3 and radf5/radb5, take an even ido
  too: terminal_merge_plan puts the radix-8 or radix-6 pass after the 3s
  and 5s, so that their ido is a multiple of 8 or 6. The pairs (i-1, i) of the i-loop then
  leave out the last element of each leg, ido-1: the leg's Nyquist term, a
  real x_j at frequency ido/2. Its twiddle, exp(-2 pi i j f/(p ido)) at
  f = ido/2, is exp(-pi i j/p), a constant, so it reads no table entry
  (rffti1_ps fills ido - 2 of the ido it reserves for a leg). The p of
  them make the output frequencies (2m+1) ido/2, m = 0 .. (p-1)/2:
    X_m = sum_j x_j exp(-pi i j (2m+1)/p),
  the others being their conjugates. In the half-complex output block of
  p*ido, Re X_m goes to element ido-1 of leg 2m and Im X_m to element 0 of
  leg 2m+1; the real X_(p-1)/2 = x0 - x1 + x2 - ... is the block's own
  Nyquist term, element ido-1 of leg p-1. Backward, unscaled like the
  other passes (p times the inverse):
    p x_j = 2 Re sum_{m < (p-1)/2} X_m exp(pi i j (2m+1)/p) + (-1)^j X_(p-1)/2.
  radf4/radb4 do the same in their ido % 2 == 0 tails.

  With an even ido a pass makes one sweep over the groups k, each doing
  its DC element, its pairs and its Nyquist terms, as radf4_groups does,
  while the group's cache lines are loaded: a separate sweep for the DC
  elements first, as FFTPACK does, made the even-ido lengths up to 9%
  slower out of cache (4 lanes, Haswell). With an odd ido a pass keeps
  FFTPACK's loops, DC elements first, and radix 3 its ido == 3 fast path.
  The *_dc and *_rows helpers are the bodies of those loops; even is a
  constant. Only terminal_merge_plan gives a 3 or a 5 an even ido, so
  without PFFFT_TERMINAL_RADIX8 the even-ido code is left out.
*/
static ALWAYS_INLINE(void) radf3_dc(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    static const float taur = (float)-0.5;                   /* cos(2*pi/3)   */
    static const float taui = (float)0.86602540378443864676; /* sin(2*pi/3) = sqrt(3)/2 */
    v4sf cr2 = VADD(VLD(cc, (k + l1) * ido), VLD(cc, (k + 2 * l1) * ido));
    VST(ch, 3 * k * ido, VADD(VLD(cc, k * ido), cr2));
    VST(ch, (3 * k + 2) * ido, SVMUL(taui, VSUB(VLD(cc, (k + l1 * 2) * ido), VLD(cc, (k + l1) * ido))));
    VST(ch, ido - 1 + (3 * k + 1) * ido, SVMADD(taur, cr2, VLD(cc, k * ido)));
}

static ALWAYS_INLINE(void) radf3_rows(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch,
                                      const float* wa1, const float* wa2, const int even) {
    static const float taur = (float)-0.5;                   /* cos(2*pi/3)   */
    static const float taui = (float)0.86602540378443864676; /* sin(2*pi/3) = sqrt(3)/2 */
    int i, ic;
    v4sf ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3, wr1, wi1, wr2, wi2;
    for (i = 2; i < ido; i += 2) {
        ic = ido - i;
        wr1 = LD_PS1(wa1[i - 2]);
        wi1 = LD_PS1(wa1[i - 1]);
        dr2 = VLD(cc, i - 1 + (k + l1) * ido);
        di2 = VLD(cc, i + (k + l1) * ido);
        VCPLXMULCONJ(dr2, di2, wr1, wi1);

        wr2 = LD_PS1(wa2[i - 2]);
        wi2 = LD_PS1(wa2[i - 1]);
        dr3 = VLD(cc, i - 1 + (k + l1 * 2) * ido);
        di3 = VLD(cc, i + (k + l1 * 2) * ido);
        VCPLXMULCONJ(dr3, di3, wr2, wi2);

        cr2 = VADD(dr2, dr3);
        ci2 = VADD(di2, di3);
        VST(ch, i - 1 + 3 * k * ido, VADD(VLD(cc, i - 1 + k * ido), cr2));
        VST(ch, i + 3 * k * ido, VADD(VLD(cc, i + k * ido), ci2));
        tr2 = SVMADD(taur, cr2, VLD(cc, i - 1 + k * ido));
        ti2 = SVMADD(taur, ci2, VLD(cc, i + k * ido));
        tr3 = SVMUL(taui, VSUB(di2, di3));
        ti3 = SVMUL(taui, VSUB(dr3, dr2));
        VST(ch, i - 1 + (3 * k + 2) * ido, VADD(tr2, tr3));
        VST(ch, ic - 1 + (3 * k + 1) * ido, VSUB(tr2, tr3));
        VST(ch, i + (3 * k + 2) * ido, VADD(ti2, ti3));
        VST(ch, ic + (3 * k + 1) * ido, VSUB(ti3, ti2));
    }
#if PFFFT_TERMINAL_RADIX8
    if (even) {
        static const float minus_taui = (float)-0.86602540378443864676; /* -sin(pi/3) */
        /* the Nyquist terms: with W = exp(-pi i/3),
           X0 = x0 + x1 W + x2 W^2 = x0 + (x1 - x2)/2 - i sin(pi/3) (x1 + x2),
           X1 = x0 - x1 + x2 */
        v4sf x0 = VLD(cc, ido - 1 + k * ido), x1 = VLD(cc, ido - 1 + (k + l1) * ido);
        v4sf x2 = VLD(cc, ido - 1 + (k + 2 * l1) * ido), d = VSUB(x2, x1);
        VST(ch, ido - 1 + 3 * k * ido, SVMADD(taur, d, x0));         /* Re X0 */
        VST(ch, (3 * k + 1) * ido, SVMUL(minus_taui, VADD(x1, x2))); /* Im X0 */
        VST(ch, ido - 1 + (3 * k + 2) * ido, VADD(x0, d));           /* X1 */
    }
#else
    (void)even;
#endif
}

static NEVER_INLINE(void)
    radf3_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1, const float* wa2) {
    static const float taur = (float)-0.5;                   /* cos(2*pi/3)   */
    static const float taui = (float)0.86602540378443864676; /* sin(2*pi/3) = sqrt(3)/2 */
    int k;
    v4sf ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3, wr1, wi1, wr2, wi2;
#if PFFFT_TERMINAL_RADIX8
    if (ido % 2 == 0) {
        for (k = 0; k < l1; k++) {
            radf3_dc(ido, l1, k, cc, ch);
            radf3_rows(ido, l1, k, cc, ch, wa1, wa2, 1);
        }
        return;
    }
#endif
    assert(ido % 2 == 1); /* else from terminal_merge_plan only */
    for (k = 0; k < l1; k++) {
        radf3_dc(ido, l1, k, cc, ch);
    }
    if (ido == 1) {
        return;
    }
    if (ido == 3) {
        /* Fast path: inner loop executes exactly once with i=2, ic=1 */
        wr1 = LD_PS1(wa1[0]);
        wi1 = LD_PS1(wa1[1]);
        wr2 = LD_PS1(wa2[0]);
        wi2 = LD_PS1(wa2[1]);
        for (k = 0; k < l1; k++) {
            dr2 = VLD(cc, 1 + (k + l1) * ido);
            di2 = VLD(cc, 2 + (k + l1) * ido);
            VCPLXMULCONJ(dr2, di2, wr1, wi1);
            dr3 = VLD(cc, 1 + (k + l1 * 2) * ido);
            di3 = VLD(cc, 2 + (k + l1 * 2) * ido);
            VCPLXMULCONJ(dr3, di3, wr2, wi2);
            cr2 = VADD(dr2, dr3);
            ci2 = VADD(di2, di3);
            VST(ch, 1 + 3 * k * ido, VADD(VLD(cc, 1 + k * ido), cr2));
            VST(ch, 2 + 3 * k * ido, VADD(VLD(cc, 2 + k * ido), ci2));
            tr2 = SVMADD(taur, cr2, VLD(cc, 1 + k * ido));
            ti2 = SVMADD(taur, ci2, VLD(cc, 2 + k * ido));
            tr3 = SVMUL(taui, VSUB(di2, di3));
            ti3 = SVMUL(taui, VSUB(dr3, dr2));
            VST(ch, 1 + (3 * k + 2) * ido, VADD(tr2, tr3));
            VST(ch, 0 + (3 * k + 1) * ido, VSUB(tr2, tr3));
            VST(ch, 2 + (3 * k + 2) * ido, VADD(ti2, ti3));
            VST(ch, 1 + (3 * k + 1) * ido, VSUB(ti3, ti2));
        }
        return;
    }
    for (k = 0; k < l1; k++) {
        radf3_rows(ido, l1, k, cc, ch, wa1, wa2, 0);
    }
} /* radf3 */

static ALWAYS_INLINE(void) radb3_dc(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    static const float taur = (float)-0.5;                    /* cos(2*pi/3)   */
    static const float taui_2 = (float)1.7320508075688772935; /* 2*sin(2*pi/3) = sqrt(3) */
    v4sf cr2, ci3, tr2;
    tr2 = VLD(cc, ido - 1 + (3 * k + 1) * ido);
    tr2 = VADD(tr2, tr2);
    cr2 = SVMADD(taur, tr2, VLD(cc, 3 * k * ido));
    VST(ch, k * ido, VADD(VLD(cc, 3 * k * ido), tr2));
    ci3 = SVMUL(taui_2, VLD(cc, (3 * k + 2) * ido));
    VST(ch, (k + l1) * ido, VSUB(cr2, ci3));
    VST(ch, (k + 2 * l1) * ido, VADD(cr2, ci3));
}

static ALWAYS_INLINE(void) radb3_rows(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch,
                                      const float* wa1, const float* wa2, const int even) {
    static const float taur = (float)-0.5;                   /* cos(2*pi/3)   */
    static const float taui = (float)0.86602540378443864676; /* sin(2*pi/3) = sqrt(3)/2 */
    int i, ic;
    v4sf ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    for (i = 2; i < ido; i += 2) {
        ic = ido - i;
        tr2 = VADD(VLD(cc, i - 1 + (3 * k + 2) * ido), VLD(cc, ic - 1 + (3 * k + 1) * ido));
        cr2 = SVMADD(taur, tr2, VLD(cc, i - 1 + 3 * k * ido));
        VST(ch, i - 1 + k * ido, VADD(VLD(cc, i - 1 + 3 * k * ido), tr2));
        ti2 = VSUB(VLD(cc, i + (3 * k + 2) * ido), VLD(cc, ic + (3 * k + 1) * ido));
        ci2 = SVMADD(taur, ti2, VLD(cc, i + 3 * k * ido));
        VST(ch, i + k * ido, VADD(VLD(cc, i + 3 * k * ido), ti2));
        cr3 = SVMUL(taui, VSUB(VLD(cc, i - 1 + (3 * k + 2) * ido), VLD(cc, ic - 1 + (3 * k + 1) * ido)));
        ci3 = SVMUL(taui, VADD(VLD(cc, i + (3 * k + 2) * ido), VLD(cc, ic + (3 * k + 1) * ido)));
        dr2 = VSUB(cr2, ci3);
        dr3 = VADD(cr2, ci3);
        di2 = VADD(ci2, cr3);
        di3 = VSUB(ci2, cr3);
        VCPLXMUL(dr2, di2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
        VST(ch, i - 1 + (k + l1) * ido, dr2);
        VST(ch, i + (k + l1) * ido, di2);
        VCPLXMUL(dr3, di3, LD_PS1(wa2[i - 2]), LD_PS1(wa2[i - 1]));
        VST(ch, i - 1 + (k + 2 * l1) * ido, dr3);
        VST(ch, i + (k + 2 * l1) * ido, di3);
    }
#if PFFFT_TERMINAL_RADIX8
    if (even) {
        static const float minus_taui_2 = (float)-1.7320508075688772935; /* -2*sin(pi/3) */
        /* the Nyquist terms from Re X0, Im X0 and X1 (see above), times 3:
           3 x0 = 2 Re X0 + X1, 3 x1, 3 x2 = +-(Re X0 - X1) - sqrt(3) Im X0 */
        v4sf r = VLD(cc, ido - 1 + 3 * k * ido), im = VLD(cc, (3 * k + 1) * ido);
        v4sf q = VLD(cc, ido - 1 + (3 * k + 2) * ido);
        v4sf d = VSUB(r, q), w = SVMUL(minus_taui_2, im);
        VST(ch, ido - 1 + k * ido, VADD(VADD(r, r), q));
        VST(ch, ido - 1 + (k + l1) * ido, VADD(d, w));
        VST(ch, ido - 1 + (k + 2 * l1) * ido, VSUB(w, d));
    }
#else
    (void)even;
#endif
}

static NEVER_INLINE(void)
    radb3_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1, const float* wa2) {
    static const float taur = (float)-0.5;                   /* cos(2*pi/3)   */
    static const float taui = (float)0.86602540378443864676; /* sin(2*pi/3) = sqrt(3)/2 */
    int k;
    v4sf ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
#if PFFFT_TERMINAL_RADIX8
    if (ido % 2 == 0) {
        for (k = 0; k < l1; k++) {
            radb3_dc(ido, l1, k, cc, ch);
            radb3_rows(ido, l1, k, cc, ch, wa1, wa2, 1);
        }
        return;
    }
#endif
    assert(ido % 2 == 1); /* else from terminal_merge_plan only */
    for (k = 0; k < l1; k++) {
        radb3_dc(ido, l1, k, cc, ch);
    }
    if (ido == 1) {
        return;
    }
    if (ido == 3) {
        /* Fast path: inner loop executes exactly once with i=2, ic=1 */
        v4sf wa1_r = LD_PS1(wa1[0]), wa1_i = LD_PS1(wa1[1]);
        v4sf wa2_r = LD_PS1(wa2[0]), wa2_i = LD_PS1(wa2[1]);
        for (k = 0; k < l1; k++) {
            tr2 = VADD(VLD(cc, 1 + (3 * k + 2) * ido), VLD(cc, 0 + (3 * k + 1) * ido));
            cr2 = SVMADD(taur, tr2, VLD(cc, 1 + 3 * k * ido));
            VST(ch, 1 + k * ido, VADD(VLD(cc, 1 + 3 * k * ido), tr2));
            ti2 = VSUB(VLD(cc, 2 + (3 * k + 2) * ido), VLD(cc, 1 + (3 * k + 1) * ido));
            ci2 = SVMADD(taur, ti2, VLD(cc, 2 + 3 * k * ido));
            VST(ch, 2 + k * ido, VADD(VLD(cc, 2 + 3 * k * ido), ti2));
            cr3 = SVMUL(taui, VSUB(VLD(cc, 1 + (3 * k + 2) * ido), VLD(cc, 0 + (3 * k + 1) * ido)));
            ci3 = SVMUL(taui, VADD(VLD(cc, 2 + (3 * k + 2) * ido), VLD(cc, 1 + (3 * k + 1) * ido)));
            dr2 = VSUB(cr2, ci3);
            dr3 = VADD(cr2, ci3);
            di2 = VADD(ci2, cr3);
            di3 = VSUB(ci2, cr3);
            VCPLXMUL(dr2, di2, wa1_r, wa1_i);
            VST(ch, 1 + (k + l1) * ido, dr2);
            VST(ch, 2 + (k + l1) * ido, di2);
            VCPLXMUL(dr3, di3, wa2_r, wa2_i);
            VST(ch, 1 + (k + 2 * l1) * ido, dr3);
            VST(ch, 2 + (k + 2 * l1) * ido, di3);
        }
        return;
    }
    for (k = 0; k < l1; k++) {
        radb3_rows(ido, l1, k, cc, ch, wa1, wa2, 0);
    }
} /* radb3 */

/* Each forward group reads ido vectors from each of four legs separated by
   stride, and writes 4 * ido consecutive vectors. The inverse exchanges
   those layouts. Handle DC, the interior pairs and (for even ido) Nyquist
   together to avoid separate sweeps over the same cache lines. */
static ALWAYS_INLINE(void) radf4_groups(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                        const float* wa2, const float* wa3) {
    static const float minus_hsqt2 = (float)-M_SQRT1_2;
    const int stride = l1 * ido;
    for (int k = 0; k < l1; ++k, cc += ido, ch += 4 * ido) {
        v4sf a0 = VLD(cc, 0), a1 = VLD(cc, stride), a2 = VLD(cc, 2 * stride), a3 = VLD(cc, 3 * stride);
        v4sf tr1 = VADD(a1, a3), tr2 = VADD(a0, a2);
        VST(ch, 0, VADD(tr1, tr2));
        VST(ch, 2 * ido - 1, VSUB(a0, a2));
        VST(ch, 2 * ido, VSUB(a3, a1));
        VST(ch, 4 * ido - 1, VSUB(tr2, tr1));

        for (int i = 2; i < ido; i += 2) {
            v4sf cr2 = VLD(cc, stride + i - 1), ci2 = VLD(cc, stride + i);
            v4sf cr3 = VLD(cc, 2 * stride + i - 1), ci3 = VLD(cc, 2 * stride + i);
            v4sf cr4 = VLD(cc, 3 * stride + i - 1), ci4 = VLD(cc, 3 * stride + i);
            VCPLXMULCONJ(cr2, ci2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
            VCPLXMULCONJ(cr3, ci3, LD_PS1(wa2[i - 2]), LD_PS1(wa2[i - 1]));
            VCPLXMULCONJ(cr4, ci4, LD_PS1(wa3[i - 2]), LD_PS1(wa3[i - 1]));

            /* The second and fourth output legs store their pairs in reverse. */
            v4sf sum_r = VADD(cr2, cr4), diff_i = VSUB(ci2, ci4);
            v4sf even_r = VADD(VLD(cc, i - 1), cr3), odd_r = VSUB(VLD(cc, i - 1), cr3);
            VST(ch, i - 1, VADD(sum_r, even_r));
            VST(ch, 4 * ido - i - 1, VSUB(even_r, sum_r));
            VST(ch, 2 * ido + i - 1, VADD(diff_i, odd_r));
            VST(ch, 2 * ido - i - 1, VSUB(odd_r, diff_i));

            v4sf sum_i = VADD(ci2, ci4), diff_r = VSUB(cr4, cr2);
            v4sf even_i = VADD(VLD(cc, i), ci3), odd_i = VSUB(VLD(cc, i), ci3);
            VST(ch, i, VADD(sum_i, even_i));
            VST(ch, 4 * ido - i, VSUB(sum_i, even_i));
            VST(ch, 2 * ido + i, VADD(diff_r, odd_i));
            VST(ch, 2 * ido - i, VSUB(diff_r, odd_i));
        }
        if (ido % 2 == 0) {
            v4sf a = VLD(cc, ido - 1 + stride), b = VLD(cc, ido - 1 + 3 * stride);
            v4sf c = VLD(cc, ido - 1), d = VLD(cc, ido - 1 + 2 * stride);
            v4sf ti = SVMUL(minus_hsqt2, VADD(a, b)), tr = SVMUL(minus_hsqt2, VSUB(b, a));
            VST(ch, ido - 1, VADD(tr, c));
            VST(ch, 3 * ido - 1, VSUB(c, tr));
            VST(ch, ido, VSUB(ti, d));
            VST(ch, 3 * ido, VADD(ti, d));
        }
    }
}

/* Specialize the twiddle-free case and the two short legs with a single
   interior pair. The compiler can hoist their twiddles out of the group loop. */
static NEVER_INLINE(void) radf4_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                   const float* wa2, const float* wa3) {
    if (ido == 1) {
        radf4_groups(1, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    if (ido == 3) {
        radf4_groups(3, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    if (ido == 4) {
        radf4_groups(4, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    radf4_groups(ido, l1, cc, ch, wa1, wa2, wa3);
}

static ALWAYS_INLINE(void) radb4_groups(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                        const float* wa2, const float* wa3) {
    static const float minus_sqrt2 = (float)-M_SQRT2, two = 2.f;
    const int stride = l1 * ido;
    for (int k = 0; k < l1; ++k, cc += 4 * ido, ch += ido) {
        v4sf a = VLD(cc, 0), b = VLD(cc, 4 * ido - 1);
        v4sf c = SVMUL(two, VLD(cc, 2 * ido)), d = SVMUL(two, VLD(cc, 2 * ido - 1));
        v4sf sum = VADD(a, b), diff = VSUB(a, b);
        VST(ch, 0, VADD(sum, d));
        VST(ch, stride, VSUB(diff, c));
        VST(ch, 2 * stride, VSUB(sum, d));
        VST(ch, 3 * stride, VADD(diff, c));

        for (int i = 2; i < ido; i += 2) {
            v4sf r0 = VLD(cc, i - 1), r1 = VLD(cc, 2 * ido - i - 1);
            v4sf r2 = VLD(cc, 2 * ido + i - 1), r3 = VLD(cc, 4 * ido - i - 1);
            v4sf i0 = VLD(cc, i), i1 = VLD(cc, 2 * ido - i);
            v4sf i2 = VLD(cc, 2 * ido + i), i3 = VLD(cc, 4 * ido - i);
            v4sf tr1 = VSUB(r0, r3), tr2 = VADD(r0, r3);
            v4sf tr3 = VADD(r2, r1), ti4 = VSUB(r2, r1);
            v4sf ti1 = VADD(i0, i3), ti2 = VSUB(i0, i3);
            v4sf tr4 = VADD(i2, i1), ti3 = VSUB(i2, i1);
            VST(ch, i - 1, VADD(tr2, tr3));
            VST(ch, i, VADD(ti2, ti3));

            v4sf cr2 = VSUB(tr1, tr4), ci2 = VADD(ti1, ti4);
            v4sf cr3 = VSUB(tr2, tr3), ci3 = VSUB(ti2, ti3);
            v4sf cr4 = VADD(tr1, tr4), ci4 = VSUB(ti1, ti4);
            VCPLXMUL(cr2, ci2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
            VST(ch, stride + i - 1, cr2);
            VST(ch, stride + i, ci2);
            VCPLXMUL(cr3, ci3, LD_PS1(wa2[i - 2]), LD_PS1(wa2[i - 1]));
            VST(ch, 2 * stride + i - 1, cr3);
            VST(ch, 2 * stride + i, ci3);
            VCPLXMUL(cr4, ci4, LD_PS1(wa3[i - 2]), LD_PS1(wa3[i - 1]));
            VST(ch, 3 * stride + i - 1, cr4);
            VST(ch, 3 * stride + i, ci4);
        }
        if (ido % 2 == 0) {
            v4sf r1 = VLD(cc, ido - 1), r3 = VLD(cc, 3 * ido - 1);
            v4sf i1 = VLD(cc, ido), i3 = VLD(cc, 3 * ido);
            v4sf tr1 = VSUB(r1, r3), tr2 = VADD(r1, r3);
            v4sf ti1 = VADD(i3, i1), ti2 = VSUB(i3, i1);
            VST(ch, ido - 1, VADD(tr2, tr2));
            VST(ch, stride + ido - 1, SVMUL(minus_sqrt2, VSUB(ti1, tr1)));
            VST(ch, 2 * stride + ido - 1, VADD(ti2, ti2));
            VST(ch, 3 * stride + ido - 1, SVMUL(minus_sqrt2, VADD(ti1, tr1)));
        }
    }
}

static NEVER_INLINE(void) radb4_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                   const float* wa2, const float* wa3) {
    if (ido == 1) {
        radb4_groups(1, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    if (ido == 3) {
        radb4_groups(3, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    if (ido == 4) {
        radb4_groups(4, l1, cc, ch, wa1, wa2, wa3);
        return;
    }
    radb4_groups(ido, l1, cc, ch, wa1, wa2, wa3);
}

/* FFTPACK's 1-based indices, all from 1 (k too): radf5 reads element i
   of leg j of group k at cc_idx(i, k, j) and writes it at ch_idx(i, j, k);
   radb5 the other way round. The offset is in the index, so that no
   pointer before the start of an array is formed. */
#define cc_idx(a_1, a_2, a_3) ((((a_3) - 1) * l1 + (a_2) - 1) * ido + (a_1) - 1)
#define ch_idx(a_1, a_2, a_3) ((((a_3) - 1) * 5 + (a_2) - 1) * ido + (a_1) - 1)

static ALWAYS_INLINE(void) radf5_dc(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    static const float ti11 = (float)0.95105651629515357212;  /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    static const float ti12 = (float)0.58778525229247312917;  /* sin(4*pi/5) */
    v4sf ci4, ci5, cr2, cr3;
    cr2 = VADD(VLD(cc, cc_idx(1, k, 5)), VLD(cc, cc_idx(1, k, 2)));
    ci5 = VSUB(VLD(cc, cc_idx(1, k, 5)), VLD(cc, cc_idx(1, k, 2)));
    cr3 = VADD(VLD(cc, cc_idx(1, k, 4)), VLD(cc, cc_idx(1, k, 3)));
    ci4 = VSUB(VLD(cc, cc_idx(1, k, 4)), VLD(cc, cc_idx(1, k, 3)));
    VST(ch, ch_idx(1, 1, k), VADD(VLD(cc, cc_idx(1, k, 1)), VADD(cr2, cr3)));
    VST(ch, ch_idx(ido, 2, k), VADD(VLD(cc, cc_idx(1, k, 1)), SVMADD(tr11, cr2, SVMUL(tr12, cr3))));
    VST(ch, ch_idx(1, 3, k), SVMADD(ti11, ci5, SVMUL(ti12, ci4)));
    VST(ch, ch_idx(ido, 4, k), VADD(VLD(cc, cc_idx(1, k, 1)), SVMADD(tr12, cr2, SVMUL(tr11, cr3))));
    VST(ch, ch_idx(1, 5, k), SVMSUB(ti11, ci4, SVMUL(ti12, ci5)));
}

static ALWAYS_INLINE(void)
    radf5_rows(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1, const float* wa2,
               const float* wa3, const float* wa4, const int even) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    static const float ti11 = (float)0.95105651629515357212;  /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    static const float ti12 = (float)0.58778525229247312917;  /* sin(4*pi/5) */
    const int idp2 = ido + 2;
    int i, ic;
    v4sf ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5, cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3,
        tr4, tr5;
    for (i = 3; i <= ido; i += 2) {
        ic = idp2 - i;
        dr2 = LD_PS1(wa1[i - 3]);
        di2 = LD_PS1(wa1[i - 2]);
        dr3 = LD_PS1(wa2[i - 3]);
        di3 = LD_PS1(wa2[i - 2]);
        dr4 = LD_PS1(wa3[i - 3]);
        di4 = LD_PS1(wa3[i - 2]);
        dr5 = LD_PS1(wa4[i - 3]);
        di5 = LD_PS1(wa4[i - 2]);
        VCPLXMULCONJ(dr2, di2, VLD(cc, cc_idx(i - 1, k, 2)), VLD(cc, cc_idx(i, k, 2)));
        VCPLXMULCONJ(dr3, di3, VLD(cc, cc_idx(i - 1, k, 3)), VLD(cc, cc_idx(i, k, 3)));
        VCPLXMULCONJ(dr4, di4, VLD(cc, cc_idx(i - 1, k, 4)), VLD(cc, cc_idx(i, k, 4)));
        VCPLXMULCONJ(dr5, di5, VLD(cc, cc_idx(i - 1, k, 5)), VLD(cc, cc_idx(i, k, 5)));
        cr2 = VADD(dr2, dr5);
        ci5 = VSUB(dr5, dr2);
        cr5 = VSUB(di2, di5);
        ci2 = VADD(di2, di5);
        cr3 = VADD(dr3, dr4);
        ci4 = VSUB(dr4, dr3);
        cr4 = VSUB(di3, di4);
        ci3 = VADD(di3, di4);
        VST(ch, ch_idx(i - 1, 1, k), VADD(VLD(cc, cc_idx(i - 1, k, 1)), VADD(cr2, cr3)));
        VST(ch, ch_idx(i, 1, k), VSUB(VLD(cc, cc_idx(i, k, 1)), VADD(ci2, ci3)));
        tr2 = VADD(VLD(cc, cc_idx(i - 1, k, 1)), SVMADD(tr11, cr2, SVMUL(tr12, cr3)));
        ti2 = VSUB(VLD(cc, cc_idx(i, k, 1)), SVMADD(tr11, ci2, SVMUL(tr12, ci3)));
        tr3 = VADD(VLD(cc, cc_idx(i - 1, k, 1)), SVMADD(tr12, cr2, SVMUL(tr11, cr3)));
        ti3 = VSUB(VLD(cc, cc_idx(i, k, 1)), SVMADD(tr12, ci2, SVMUL(tr11, ci3)));
        tr5 = SVMADD(ti11, cr5, SVMUL(ti12, cr4));
        ti5 = SVMADD(ti11, ci5, SVMUL(ti12, ci4));
        tr4 = SVMSUB(ti11, cr4, SVMUL(ti12, cr5));
        ti4 = SVMSUB(ti11, ci4, SVMUL(ti12, ci5));
        VST(ch, ch_idx(i - 1, 3, k), VSUB(tr2, tr5));
        VST(ch, ch_idx(ic - 1, 2, k), VADD(tr2, tr5));
        VST(ch, ch_idx(i, 3, k), VADD(ti2, ti5));
        VST(ch, ch_idx(ic, 2, k), VSUB(ti5, ti2));
        VST(ch, ch_idx(i - 1, 5, k), VSUB(tr3, tr4));
        VST(ch, ch_idx(ic - 1, 4, k), VADD(tr3, tr4));
        VST(ch, ch_idx(i, 5, k), VADD(ti3, ti4));
        VST(ch, ch_idx(ic, 4, k), VSUB(ti4, ti3));
    }
#if PFFFT_TERMINAL_RADIX8
    if (even) {
        static const float minus_ti11 = (float)-0.95105651629515357212; /* -sin(2*pi/5) */
        /* the Nyquist terms (see above radf3_dc): with W = exp(-pi i/5),
           X0 = sum_j x_j W^j and X1 = sum_j x_j W^(3j) are
             Re X0 = x0 - cos(pi/5) (x4 - x1) + cos(2*pi/5) (x2 - x3),
             Im X0 = -sin(pi/5) (x1 + x4) - sin(2*pi/5) (x2 + x3),
             Re X1 = x0 + cos(2*pi/5) (x4 - x1) - cos(pi/5) (x2 - x3),
             Im X1 = sin(pi/5) (x2 + x3) - sin(2*pi/5) (x1 + x4),
           and X2 = x0 - x1 + x2 - x3 + x4; cos(pi/5) = -tr12, sin(pi/5) = ti12 */
        v4sf x0 = VLD(cc, cc_idx(ido, k, 1));
        v4sf a = VSUB(VLD(cc, cc_idx(ido, k, 5)), VLD(cc, cc_idx(ido, k, 2))); /* x4 - x1 */
        v4sf b = VSUB(VLD(cc, cc_idx(ido, k, 3)), VLD(cc, cc_idx(ido, k, 4))); /* x2 - x3 */
        v4sf c = VADD(VLD(cc, cc_idx(ido, k, 2)), VLD(cc, cc_idx(ido, k, 5))); /* x1 + x4 */
        v4sf d = VADD(VLD(cc, cc_idx(ido, k, 3)), VLD(cc, cc_idx(ido, k, 4))); /* x2 + x3 */
        VST(ch, ch_idx(ido, 1, k), VADD(x0, SVMADD(tr12, a, SVMUL(tr11, b))));
        VST(ch, ch_idx(1, 2, k), SVMSUB(ti12, c, SVMUL(minus_ti11, d)));
        VST(ch, ch_idx(ido, 3, k), VADD(x0, SVMADD(tr11, a, SVMUL(tr12, b))));
        VST(ch, ch_idx(1, 4, k), SVMSUB(ti11, c, SVMUL(ti12, d)));
        VST(ch, ch_idx(ido, 5, k), VADD(x0, VADD(a, b)));
    }
#else
    (void)even;
#endif
}

static NEVER_INLINE(void) radf5_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                   const float* wa2, const float* wa3, const float* wa4) {
    int k;
#if PFFFT_TERMINAL_RADIX8
    if (ido % 2 == 0) {
        for (k = 1; k <= l1; ++k) {
            radf5_dc(ido, l1, k, cc, ch);
            radf5_rows(ido, l1, k, cc, ch, wa1, wa2, wa3, wa4, 1);
        }
        return;
    }
#endif
    assert(ido % 2 == 1); /* else from terminal_merge_plan only */
    for (k = 1; k <= l1; ++k) {
        radf5_dc(ido, l1, k, cc, ch);
    }
    if (ido == 1) {
        return;
    }
    for (k = 1; k <= l1; ++k) {
        radf5_rows(ido, l1, k, cc, ch, wa1, wa2, wa3, wa4, 0);
    }
} /* radf5 */

#undef cc_idx
#undef ch_idx

#define cc_idx(a_1, a_2, a_3) ((((a_3) - 1) * 5 + (a_2) - 1) * ido + (a_1) - 1)
#define ch_idx(a_1, a_2, a_3) ((((a_3) - 1) * l1 + (a_2) - 1) * ido + (a_1) - 1)

static ALWAYS_INLINE(void) radb5_dc(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    static const float ti11 = (float)0.95105651629515357212;  /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    static const float ti12 = (float)0.58778525229247312917;  /* sin(4*pi/5) */
    v4sf ci4, ci5, cr2, cr3, ti4, ti5, tr2, tr3;
    ti5 = VADD(VLD(cc, cc_idx(1, 3, k)), VLD(cc, cc_idx(1, 3, k)));
    ti4 = VADD(VLD(cc, cc_idx(1, 5, k)), VLD(cc, cc_idx(1, 5, k)));
    tr2 = VADD(VLD(cc, cc_idx(ido, 2, k)), VLD(cc, cc_idx(ido, 2, k)));
    tr3 = VADD(VLD(cc, cc_idx(ido, 4, k)), VLD(cc, cc_idx(ido, 4, k)));
    VST(ch, ch_idx(1, k, 1), VADD(VLD(cc, cc_idx(1, 1, k)), VADD(tr2, tr3)));
    cr2 = VADD(VLD(cc, cc_idx(1, 1, k)), SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
    cr3 = VADD(VLD(cc, cc_idx(1, 1, k)), SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
    ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
    ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
    VST(ch, ch_idx(1, k, 2), VSUB(cr2, ci5));
    VST(ch, ch_idx(1, k, 3), VSUB(cr3, ci4));
    VST(ch, ch_idx(1, k, 4), VADD(cr3, ci4));
    VST(ch, ch_idx(1, k, 5), VADD(cr2, ci5));
}

static ALWAYS_INLINE(void)
    radb5_rows(int ido, int l1, int k, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1, const float* wa2,
               const float* wa3, const float* wa4, const int even) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    static const float ti11 = (float)0.95105651629515357212;  /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    static const float ti12 = (float)0.58778525229247312917;  /* sin(4*pi/5) */
    const int idp2 = ido + 2;
    int i, ic;
    v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3,
        tr4, tr5;
    for (i = 3; i <= ido; i += 2) {
        ic = idp2 - i;
        ti5 = VADD(VLD(cc, cc_idx(i, 3, k)), VLD(cc, cc_idx(ic, 2, k)));
        ti2 = VSUB(VLD(cc, cc_idx(i, 3, k)), VLD(cc, cc_idx(ic, 2, k)));
        ti4 = VADD(VLD(cc, cc_idx(i, 5, k)), VLD(cc, cc_idx(ic, 4, k)));
        ti3 = VSUB(VLD(cc, cc_idx(i, 5, k)), VLD(cc, cc_idx(ic, 4, k)));
        tr5 = VSUB(VLD(cc, cc_idx(i - 1, 3, k)), VLD(cc, cc_idx(ic - 1, 2, k)));
        tr2 = VADD(VLD(cc, cc_idx(i - 1, 3, k)), VLD(cc, cc_idx(ic - 1, 2, k)));
        tr4 = VSUB(VLD(cc, cc_idx(i - 1, 5, k)), VLD(cc, cc_idx(ic - 1, 4, k)));
        tr3 = VADD(VLD(cc, cc_idx(i - 1, 5, k)), VLD(cc, cc_idx(ic - 1, 4, k)));
        VST(ch, ch_idx(i - 1, k, 1), VADD(VLD(cc, cc_idx(i - 1, 1, k)), VADD(tr2, tr3)));
        VST(ch, ch_idx(i, k, 1), VADD(VLD(cc, cc_idx(i, 1, k)), VADD(ti2, ti3)));
        cr2 = VADD(VLD(cc, cc_idx(i - 1, 1, k)), SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
        ci2 = VADD(VLD(cc, cc_idx(i, 1, k)), SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
        cr3 = VADD(VLD(cc, cc_idx(i - 1, 1, k)), SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
        ci3 = VADD(VLD(cc, cc_idx(i, 1, k)), SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
        cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
        ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
        cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
        ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
        dr3 = VSUB(cr3, ci4);
        dr4 = VADD(cr3, ci4);
        di3 = VADD(ci3, cr4);
        di4 = VSUB(ci3, cr4);
        dr5 = VADD(cr2, ci5);
        dr2 = VSUB(cr2, ci5);
        di5 = VSUB(ci2, cr5);
        di2 = VADD(ci2, cr5);
        VCPLXMUL(dr2, di2, LD_PS1(wa1[i - 3]), LD_PS1(wa1[i - 2]));
        VCPLXMUL(dr3, di3, LD_PS1(wa2[i - 3]), LD_PS1(wa2[i - 2]));
        VCPLXMUL(dr4, di4, LD_PS1(wa3[i - 3]), LD_PS1(wa3[i - 2]));
        VCPLXMUL(dr5, di5, LD_PS1(wa4[i - 3]), LD_PS1(wa4[i - 2]));

        VST(ch, ch_idx(i - 1, k, 2), dr2);
        VST(ch, ch_idx(i, k, 2), di2);
        VST(ch, ch_idx(i - 1, k, 3), dr3);
        VST(ch, ch_idx(i, k, 3), di3);
        VST(ch, ch_idx(i - 1, k, 4), dr4);
        VST(ch, ch_idx(i, k, 4), di4);
        VST(ch, ch_idx(i - 1, k, 5), dr5);
        VST(ch, ch_idx(i, k, 5), di5);
    }
#if PFFFT_TERMINAL_RADIX8
    if (even) {
        static const float minus_ti11 = (float)-0.95105651629515357212; /* -sin(2*pi/5) */
        /* the Nyquist terms from X0, X1 and X2 (see radf5_rows), times 5:
             5 x0 = 2 Re X0 + 2 Re X1 + X2,
             5 x1, 5 x4 = -+(X2 - 2 cos(pi/5) Re X0 + 2 cos(2*pi/5) Re X1)
                          - 2 sin(pi/5) Im X0 - 2 sin(2*pi/5) Im X1,
             5 x2, 5 x3 = +-(X2 + 2 cos(2*pi/5) Re X0 - 2 cos(pi/5) Re X1)
                          - 2 sin(2*pi/5) Im X0 + 2 sin(pi/5) Im X1 */
        v4sf q = VLD(cc, cc_idx(ido, 5, k));
        v4sf r0 = VADD(VLD(cc, cc_idx(ido, 1, k)), VLD(cc, cc_idx(ido, 1, k)));
        v4sf i0 = VADD(VLD(cc, cc_idx(1, 2, k)), VLD(cc, cc_idx(1, 2, k)));
        v4sf r1 = VADD(VLD(cc, cc_idx(ido, 3, k)), VLD(cc, cc_idx(ido, 3, k)));
        v4sf i1 = VADD(VLD(cc, cc_idx(1, 4, k)), VLD(cc, cc_idx(1, 4, k)));
        v4sf p1 = VADD(q, SVMADD(tr12, r0, SVMUL(tr11, r1)));
        v4sf p2 = VADD(q, SVMADD(tr11, r0, SVMUL(tr12, r1)));
        v4sf s1 = SVMSUB(ti12, i0, SVMUL(minus_ti11, i1));
        v4sf s2 = SVMSUB(ti11, i0, SVMUL(ti12, i1));
        VST(ch, ch_idx(ido, k, 1), VADD(q, VADD(r0, r1)));
        VST(ch, ch_idx(ido, k, 2), VSUB(s1, p1));
        VST(ch, ch_idx(ido, k, 3), VADD(s2, p2));
        VST(ch, ch_idx(ido, k, 4), VSUB(s2, p2));
        VST(ch, ch_idx(ido, k, 5), VADD(s1, p1));
    }
#else
    (void)even;
#endif
}

static NEVER_INLINE(void) radb5_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch, const float* wa1,
                                   const float* wa2, const float* wa3, const float* wa4) {
    int k;
#if PFFFT_TERMINAL_RADIX8
    if (ido % 2 == 0) {
        for (k = 1; k <= l1; ++k) {
            radb5_dc(ido, l1, k, cc, ch);
            radb5_rows(ido, l1, k, cc, ch, wa1, wa2, wa3, wa4, 1);
        }
        return;
    }
#endif
    assert(ido % 2 == 1); /* else from terminal_merge_plan only */
    for (k = 1; k <= l1; ++k) {
        radb5_dc(ido, l1, k, cc, ch);
    }
    if (ido == 1) {
        return;
    }
    for (k = 1; k <= l1; ++k) {
        radb5_rows(ido, l1, k, cc, ch, wa1, wa2, wa3, wa4, 0);
    }
} /* radb5 */

#undef cc_idx
#undef ch_idx

#if PFFFT_TERMINAL_RADIX8
/*
  Radix-8 real passes at ido == 1 (no twiddles), from terminal_merge_plan:
  the first pass of the forward transform and the last pass of the backward
  one. They replace the radix-2 pass, the one with the longest twiddle
  table, and a radix-4 pass: both are dropped and the radix-8 pass runs at
  ido == 1, so every pass in between runs with other l1 and ido (the 3s
  and 5s with an even one); the transform has one pass over the data
  fewer.

  forward:  X_m = sum_j a_j exp(-2 pi i j m/8), stored half-complex as
            R0 R1 I1 R2 I2 R3 I3 R4
  backward: 8 a_j from the same layout (unscaled, as FFTPACK's passes)
*/
static NEVER_INLINE(void) radf8_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    const float c8 = (float)M_SQRT1_2; /* 1/sqrt(2) */
    const v4sf k8 = LD_PS1(c8);        /* named: see passf8_ps */
    int k;
    assert(ido == 1);
    (void)ido; /* otherwise unused when assertions are disabled */
    for (k = 0; k < l1; ++k, ++cc, ch += 8) {
        v4sf a0 = VLD(cc, 0), a4 = VLD(cc, 4 * l1);
        v4sf a1 = VLD(cc, l1), a5 = VLD(cc, 5 * l1);
        v4sf a2 = VLD(cc, 2 * l1), a6 = VLD(cc, 6 * l1);
        v4sf a3 = VLD(cc, 3 * l1), a7 = VLD(cc, 7 * l1);
        v4sf s0 = VADD(a0, a4), d0 = VSUB(a0, a4);
        v4sf s1 = VADD(a1, a5), d1 = VSUB(a1, a5);
        v4sf s2 = VADD(a2, a6), d2 = VSUB(a2, a6), nd2 = VSUB(a6, a2);
        v4sf s3 = VADD(a3, a7), d3 = VSUB(a3, a7);
        v4sf e0 = VADD(s0, s2), e1 = VADD(s1, s3);
        v4sf u = VSUB(d1, d3), v = VADD(d1, d3);
        VST(ch, 0, VADD(e0, e1));      /* R0 */
        VST(ch, 1, VMADD(k8, u, d0));  /* R1 = d0 + u/sqrt2 */
        VST(ch, 2, VMSUB(k8, v, nd2)); /* I1 = -d2 - v/sqrt2 */
        VST(ch, 3, VSUB(s0, s2));      /* R2 */
        VST(ch, 4, VSUB(s3, s1));      /* I2 */
        VST(ch, 5, VMSUB(k8, u, d0));  /* R3 = d0 - u/sqrt2 */
        VST(ch, 6, VMSUB(k8, v, d2));  /* I3 = d2 - v/sqrt2 */
        VST(ch, 7, VSUB(e0, e1));      /* R4 */
    }
} /* radf8 */

static NEVER_INLINE(void) radb8_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    const float c2 = 2.f, csq2 = (float)M_SQRT2;
    const v4sf two = LD_PS1(c2), sq2 = LD_PS1(csq2); /* named: see passf8_ps */
    int k;
    assert(ido == 1);
    (void)ido; /* otherwise unused when assertions are disabled */
    for (k = 0; k < l1; ++k, cc += 8, ++ch) {
        v4sf r0 = VLD(cc, 0), r1 = VLD(cc, 1), i1 = VLD(cc, 2), r2 = VLD(cc, 3);
        v4sf i2 = VLD(cc, 4), r3 = VLD(cc, 5), i3 = VLD(cc, 6), r4 = VLD(cc, 7);
        v4sf p = VADD(r0, r4), q = VSUB(r0, r4);
        v4sf a = VADD(r1, r3), b = VSUB(r1, r3), c = VSUB(i3, i1), d = VADD(i1, i3);
        v4sf e0 = VMADD(two, r2, p), e2 = VMSUB(two, r2, p);
        v4sf e1 = VMSUB(two, i2, q), e3 = VMADD(two, i2, q);
        v4sf f1 = VSUB(b, d), f3 = VADD(b, d);
        VST(ch, 0, VMADD(two, a, e0));
        VST(ch, 4 * l1, VMSUB(two, a, e0));
        VST(ch, 2 * l1, VMADD(two, c, e2));
        VST(ch, 6 * l1, VMSUB(two, c, e2));
        VST(ch, 1 * l1, VMADD(sq2, f1, e1));
        VST(ch, 5 * l1, VMSUB(sq2, f1, e1));
        VST(ch, 3 * l1, VMSUB(sq2, f3, e3));
        VST(ch, 7 * l1, VMADD(sq2, f3, e3));
    }
} /* radb8 */

/*
  Real radix-6 passes at ido == 1, like radf8_ps and radb8_ps.
  2 and 3 are coprime, so the 6-point DFT is a 2-point DFT of each of the
  pairs (x0, x3), (x2, x5), (x4, x1) and a 3-point DFT of their sums and
  one of their differences, with no twiddles in between (Good-Thomas:
  input index 3 j1 + 2 j2, output index 3 k1 + 4 k2, mod 6). The sums give
  X0, X4, X2, the differences X3, X1, X5, of which X0 .. X3 are stored.

  forward:  half-complex R0 R1 I1 R2 I2 R3
  backward: 6 a_j from the same layout (unscaled)
*/
static NEVER_INLINE(void) radf6_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    const float ctaur = -0.5f, ctaui = (float)0.86602540378443864676; /* cos, sin(2*pi/3) */
    const v4sf taur = LD_PS1(ctaur), taui = LD_PS1(ctaui);            /* named: see passf8_ps */
    int k;
    assert(ido == 1);
    (void)ido; /* otherwise unused when assertions are disabled */
    for (k = 0; k < l1; ++k, ++cc, ch += 6) {
        v4sf x0 = VLD(cc, 0), x1 = VLD(cc, l1), x2 = VLD(cc, 2 * l1);
        v4sf x3 = VLD(cc, 3 * l1), x4 = VLD(cc, 4 * l1), x5 = VLD(cc, 5 * l1);
        v4sf u0 = VADD(x0, x3), u1 = VADD(x2, x5), u2 = VADD(x4, x1);
        v4sf v0 = VSUB(x0, x3), v1 = VSUB(x2, x5), v2 = VSUB(x4, x1);
        v4sf su = VADD(u1, u2), sv = VADD(v1, v2);
        VST(ch, 0, VADD(u0, su));             /* R0 */
        VST(ch, 1, VMADD(taur, sv, v0));      /* R1 */
        VST(ch, 2, VMUL(taui, VSUB(v2, v1))); /* I1 */
        VST(ch, 3, VMADD(taur, su, u0));      /* R2 */
        VST(ch, 4, VMUL(taui, VSUB(u1, u2))); /* I2 */
        VST(ch, 5, VADD(v0, sv));             /* R3 */
    }
} /* radf6 */

static NEVER_INLINE(void) radb6_ps(int ido, int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT ch) {
    const float c2 = 2.f, csq3 = (float)1.7320508075688772935; /* 2, sqrt(3) */
    const v4sf two = LD_PS1(c2), sq3 = LD_PS1(csq3);           /* named: see passf8_ps */
    int k;
    assert(ido == 1);
    (void)ido; /* otherwise unused when assertions are disabled */
    for (k = 0; k < l1; ++k, cc += 6, ++ch) {
        v4sf x0 = VLD(cc, 0), r1 = VLD(cc, 1), i1 = VLD(cc, 2);
        v4sf r2 = VLD(cc, 3), i2 = VLD(cc, 4), x3 = VLD(cc, 5);
        v4sf u0 = VMADD(two, r2, x0), a = VSUB(x0, r2), b = VMUL(sq3, i2);
        v4sf v0 = VMADD(two, r1, x3), c = VSUB(x3, r1), d = VMUL(sq3, i1);
        v4sf u1 = VADD(a, b), u2 = VSUB(a, b), v1 = VSUB(c, d), v2 = VADD(c, d);
        VST(ch, 0, VADD(u0, v0));
        VST(ch, 3 * l1, VSUB(u0, v0));
        VST(ch, 2 * l1, VADD(u1, v1));
        VST(ch, 5 * l1, VSUB(u1, v1));
        VST(ch, 4 * l1, VADD(u2, v2));
        VST(ch, 1 * l1, VSUB(u2, v2));
    }
} /* radb6 */
#endif /* PFFFT_TERMINAL_RADIX8 */

static NEVER_INLINE(v4sf*)
    rfftf1_ps(int n, const v4sf* input_readonly, v4sf* work1, v4sf* work2, const float* wa, const int* ifac) {
    v4sf* in = (v4sf*)input_readonly;
    v4sf* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l1 = n, ido = 1;
    assert(in != out && work1 != work2);
    for (k1 = 1; k1 <= nf; ++k1) {
        int kh = nf - k1;
        int ip = ifac[kh + 2];
        /* For this stage n = l1 * ip * ido. Earlier stages reserve
           (radix - 1) * inner floats each, totalling n - ip * ido. */
        const int next_ido = ip * ido;
        const float* w = wa + (n - next_ido);
#if PFFFT_TERMINAL_RADIX8
        /* radix 8 and 6 are tested in front of the switch rather than as
       cases of it. check_isa_baseline rejects an indirect jump in 8-lane
       code while the upper ymm halves may be dirty, unless it proves the
       jump a local jump table. Here they are clean: the driver has no
       256-bit code of its own, and every pass it calls, radix 3 and 5
       included, is a NEVER_INLINE function that clears them after its
       256-bit code (checked in GCC and Clang Release builds), and the
       drivers are entered with them clean (the audit follows the state
       through their callers). A jump table
       passes the audit either way: Clang folds the ifs and the switch into
       one, GCC makes one when radix 8 is a fifth case and none as
       written. */
        if (ip == 8) {
            l1 /= 8;
            radf8_ps(ido, l1, in, out);
        } else if (ip == 6) {
            l1 /= 6;
            radf6_ps(ido, l1, in, out);
        } else
#endif
            switch (ip) {
                case 5:
                    l1 /= 5;
                    radf5_ps(ido, l1, in, out, w, w + ido, w + 2 * ido, w + 3 * ido);
                    break;
                case 4:
                    l1 /= 4;
                    radf4_ps(ido, l1, in, out, w, w + ido, w + 2 * ido);
                    break;
                case 3:
                    l1 /= 3;
                    radf3_ps(ido, l1, in, out, w, w + ido);
                    break;
                case 2:
                    l1 /= 2;
                    radf2_ps(ido, l1, in, out, w);
                    break;
                default: assert(0); break;
            }
        ido = next_ido;
        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
} /* rfftf1 */

static NEVER_INLINE(v4sf*)
    rfftb1_ps(int n, const v4sf* input_readonly, v4sf* work1, v4sf* work2, const float* wa, const int* ifac) {
    v4sf* in = (v4sf*)input_readonly;
    v4sf* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l1 = 1, ido = n;
    assert(in != out);
    for (k1 = 1; k1 <= nf; k1++) {
        int ip = ifac[k1 + 1];
        /* Before division, ido spans all ip legs of this stage. */
        const float* w = wa + (n - ido);
#if PFFFT_TERMINAL_RADIX8
        /* in front of the switch, see rfftf1_ps */
        if (ip == 8) {
            ido /= 8;
            radb8_ps(ido, l1, in, out);
        } else if (ip == 6) {
            ido /= 6;
            radb6_ps(ido, l1, in, out);
        } else
#endif
            switch (ip) {
                case 5:
                    ido /= 5;
                    radb5_ps(ido, l1, in, out, w, w + ido, w + 2 * ido, w + 3 * ido);
                    break;
                case 4:
                    ido /= 4;
                    radb4_ps(ido, l1, in, out, w, w + ido, w + 2 * ido);
                    break;
                case 3:
                    ido /= 3;
                    radb3_ps(ido, l1, in, out, w, w + ido);
                    break;
                case 2:
                    ido /= 2;
                    radb2_ps(ido, l1, in, out, w);
                    break;
                default: assert(0); break;
            }
        l1 *= ip;

        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
}

#ifndef IFAC_MAX_SIZE
#define IFAC_MAX_SIZE 25 /* max number of integer factors for the decomposition, +2 */
#endif
static int
decompose(int n, int* ifac, const int* ntryh) {
    int nl = n, nf = 0, i, j = 0;
    for (j = 0; ntryh[j]; ++j) {
        int ntry = ntryh[j];
        while (nl != 1) {
            int nq = nl / ntry;
            int nr = nl - ntry * nq;
            if (nr == 0) {
                assert(2 + nf < IFAC_MAX_SIZE);
                ifac[2 + nf++] = ntry;
                nl = nq;
                if (ntry == 2 && nf != 1) {
                    for (i = 2; i <= nf; ++i) {
                        int ib = nf - i + 2;
                        ifac[ib + 1] = ifac[ib];
                    }
                    ifac[2] = 2;
                }
            } else {
                break;
            }
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    return nf;
}

/*
  cos and sin of 2*pi*m/n for 0 <= m < n. The angle is reduced exactly in
  integers to [0, pi/4] and evaluated in long double, so every factor is
  correctly rounded or nearly so (at most ~1 ulp where long double is
  double), and the symmetric factors (+-1, 0, cos = sin at pi/4, ...) are
  exact. Repeated rounding of 2*pi/n * m in the working precision instead
  costs up to ~5 eps per factor in float.
*/
static void
twiddle_cos_sin(int m, int n, float* c, float* s) {
    const long double pi_2 = 1.570796326794896619231321691639751442L;
    long q = (4L * m) / n;   /* quadrant */
    long r = 4L * m - q * n; /* angle in the quadrant = pi/2 * r/n */
    long double x, cx, sx, t;
    if (2 * r <= n) {
        x = pi_2 * (long double)r / n;
        cx = cosl(x);
        sx = sinl(x);
    } else { /* pi/2 - angle is in [0, pi/4) */
        x = pi_2 * (long double)(n - r) / n;
        cx = sinl(x);
        sx = cosl(x);
    }
    switch (q & 3) {
        case 1:
            t = cx;
            cx = -sx;
            sx = t;
            break;
        case 2:
            cx = -cx;
            sx = -sx;
            break;
        case 3:
            t = cx;
            cx = sx;
            sx = -t;
            break;
        default: break;
    }
    *c = (float)cx;
    *s = (float)sx;
}

#if PFFFT_TERMINAL_RADIX8
/*
  The plan rule of PFFFT_TERMINAL_RADIX8, applied by rffti1_ps and
  cffti1_ps to the factor list decompose() made: no stand-alone radix-2
  pass. If the list starts with a 2 (decompose puts a 2 first), the 2 and
  a partner are dropped and their product is appended: with radix6 and a
  3 in the list, the 2 and a 3 make a 6; else, with a 4, the 2 and a 4
  make an 8. ifac[1] is updated. Returns the number of factors.

  The last factor is the pass with one element per leg: the last pass of
  a complex transform in both directions, the first pass of the real
  forward transform and the last of the real backward one. There the
  radix-8 and radix-6 passes (passf8_ps, passf6_ps, radf8_ps, radb8_ps,
  radf6_ps, radb6_ps) need no twiddles. Both merges leave as many passes.
  Complex lists 2, 5..., 3, 3..., 4... become 5..., 3..., 4..., 6, with
  one 3 fewer (cffti1_factors asks for radix 6 at every length), and
  without a 3, 2, 5..., 4, 4... become 5..., 4..., 8, with one 4 fewer.
  The radix-6 merge drops a twiddled radix-3 pass where the radix-8 one
  drops a radix-4 pass. Against the radix-8 merge (complex transforms,
  every such length up to N = 393216) it took, with 4 lanes, 1 to 8% less
  time in cache and up to 2.5% less out of cache on Haswell and Zen 3,
  and with 8 lanes on Zen 3 from 3% less to 2% more in cache (up to 7%
  less at N = 12288) and 1 to 12% less out of cache; single lengths were
  slower by up to 3% (Haswell: N = 6144 and 18432, in every transform
  mode). A list 4, 4...,
  6 (N/SIMD_SZ = 3 * 2^(2k+1)) starts with radix-4 passes and so takes
  the depth-first schedule above its threshold, as powers of two do: with
  8 lanes (Zen 3) 2 to 8% less time than the other schedule at N = 49152
  and 196608; with 4 lanes at N = 1572864 13 to 21% less on Haswell, and
  on Zen 3 the same forward and 7% more backward. Real lists 2, 4,
  4..., 3..., 5... become 4..., 3..., 5..., 8, or, with radix6 and a 3,
  4, 4..., 3..., 5..., 6 with one 3 fewer (rffti1_ps asks for it up to
  PFFFT_REAL_RADIX6_MAX_BYTES or ..._WIDE, by lane count; see
  pffft_pass_config.h). The 3s and 5s,
  which FFTPACK's order {4,2,3,5} puts last so that their ido is odd,
  then get an even ido (a multiple of 8 or 6), which radf3/radb3 and
  radf5/radb5 handle. The other factors keep their order; no other order
  of the 4s, 3s and 5s before the 8 was faster beyond the noise (4 lanes,
  Haswell, N = 480 to 192000), and several were 5 to 10% slower out of
  cache. The twiddle table needs no change: its offsets add up (ip-1)*ido
  per pass in any order, and an even ido reads ido-2 entries per leg,
  which is what rffti1_ps fills. A list without a 4 (one lane only:
  2, 3..., 5...) takes the radix-6 merge too.
*/
static int
terminal_merge_plan(int* ifac, int radix6) {
    int nf = ifac[1], j = 3, k, merged = 6;
    if (nf < 2 || ifac[2] != 2) {
        return nf;
    }
    while (radix6 && j <= nf + 1 && ifac[j] != 3) {
        ++j;
    }
    if (!radix6 || j > nf + 1) {
        merged = 8;
        j = 3;
        while (j <= nf + 1 && ifac[j] != 4) {
            ++j;
        }
        if (j > nf + 1) {
            return nf;
        }
    }
    /* drop ifac[2], the 2, and ifac[j], the first 3 or 4 */
    for (k = 2; k < j - 1; ++k) {
        ifac[k] = ifac[k + 1];
    }
    for (k = j - 1; k < nf; ++k) {
        ifac[k] = ifac[k + 2];
    }
    ifac[nf] = merged;
    ifac[1] = --nf;
    return nf;
}
#endif /* PFFFT_TERMINAL_RADIX8 */

static void
rffti1_ps(int n, float* wa, int* ifac) {
    static const int ntryh[] = {4, 2, 3, 5, 0};
    int k1, j, ii;

#if PFFFT_TERMINAL_RADIX8
    int nf;
    decompose(n, ifac, ntryh);
    /* radix 6 up to the limit of this backend's lane count (bytes of an
       array: n vectors, 4 N bytes, with any number of lanes) */
    nf = terminal_merge_plan(ifac, (unsigned long long)n * sizeof(v4sf)
                                       <= (sizeof(v4sf) == 8 * sizeof(float)
                                               ? (unsigned long long)(PFFFT_REAL_RADIX6_MAX_BYTES_WIDE)
                                               : (unsigned long long)(PFFFT_REAL_RADIX6_MAX_BYTES)));
#else
    int nf = decompose(n, ifac, ntryh);
#endif
    int is = 0;
    int nfm1 = nf - 1;
    int l1 = 1;
    for (k1 = 1; k1 <= nfm1; k1++) {
        int ip = ifac[k1 + 1];
        int ld = 0;
        int l2 = l1 * ip;
        int ido = n / l2;
        int ipm = ip - 1;
        for (j = 1; j <= ipm; ++j) {
            int i = is, fi = 0;
            ld += l1;
            for (ii = 3; ii <= ido; ii += 2) {
                i += 2;
                fi += 1;
                twiddle_cos_sin(fi * ld, n, &wa[i - 2], &wa[i - 1]); /* fi*ld < n */
            }
            is += ido;
        }
        l1 = l2;
    }
} /* rffti1 */

/* the factors of the complex plan of n, as cffti1_ps stores them in ifac;
   returns their number */
static int
cffti1_factors(int n, int* ifac) {
#if PFFFT_TERMINAL_RADIX8
    static const int ntryh[] = {5, 3, 4, 2, 0};
    decompose(n, ifac, ntryh);
    /* radix 6 wherever there is a 3 (see terminal_merge_plan) */
    return terminal_merge_plan(ifac, 1);
#else
    static const int ntryh_large[] = {5, 3, 8, 4, 2, 0};
    static const int ntryh_default[] = {5, 3, 4, 2, 0};
    return decompose(n, ifac, (n == 8) ? ntryh_large : ntryh_default);
#endif
}

static void
cffti1_ps(int n, float* wa, int* ifac) {
    /* Radix 8 is used only as the pass with one element per leg, the last
     one, where it needs no twiddles. As a twiddled pass in an earlier
     position it is slower: on x86-64, the passes alone (without the
     cross-lane step) of an 8-lane complex transform of N = 16384, per-lane
     length n = 2048, took 1.47 times as long with 8,8,8,4 as with
     2,4,4,4,4,4, and the arm64 measurements below found the same. Two
     regimes:

     PFFFT_TERMINAL_RADIX8 == 0: radix 8 only for n == 8 (the Ncvec==8
     case, N=32 complex on 4 lanes), where a single dedicated pass is
     guaranteed. Measured on Mac/arm64, Android/arm64-v8a and iPhone/arm64
     (20-rep sustained, Mann-Whitney U + Hodges-Lehmann shift): Ncvec==8
     is a clear win (-2% to -21%) on every platform/precision; Ncvec==16
     regressed double precision by +4% to +6% on every platform even when
     restricted-in, and was mixed-to-negative for float, so it was
     excluded and the radix-16 codelet (passf16_ps) removed entirely --
     it would otherwise be unreachable at every N. Larger multi-pass
     sizes regress severely (+15% to +27% at N=16384) if this gate is
     lifted.

     PFFFT_TERMINAL_RADIX8 == 1: the default try-list for every n, then
     terminal_merge_plan: a list 2, ..., 3, ... becomes ..., 6 with one 3
     fewer, else 2, ..., 4 becomes ..., 8 (n == 8 still gives the single
     pass 8), which saves one sweep over the data. */
    int k1, j, ii;
    int nf = cffti1_factors(n, ifac);

    int i = 1;
    int l1 = 1;
    for (k1 = 1; k1 <= nf; k1++) {
        int ip = ifac[k1 + 1];
        int ld = 0;
        int l2 = l1 * ip;
        int ido = n / l2;
        int idot = ido + ido + 2;
        int ipm = ip - 1;
        for (j = 1; j <= ipm; j++) {
            int i1 = i, fi = 0;
            wa[i - 1] = 1;
            wa[i] = 0;
            ld += l1;
            for (ii = 4; ii <= idot; ii += 2) {
                i += 2;
                fi += 1;
                twiddle_cos_sin(fi * ld, n, &wa[i - 1], &wa[i]); /* fi*ld < n */
            }
            if (ip > 16) {
                wa[i1 - 1] = wa[i - 1];
                wa[i1] = wa[i];
            }
        }
        l1 = l2;
    }
} /* cffti1 */

#ifndef PF_FUSED_LAYOUT /* superseded by cfftf1_il_ps / cfftf1_tail_ol_ps below */
static v4sf*
cfftf1_ps(int n, const v4sf* input_readonly, v4sf* work1, v4sf* work2, const float* wa, const int* ifac, int isign) {
    v4sf* in = (v4sf*)input_readonly;
    v4sf* out = (in == work2 ? work1 : work2);
    int nf = ifac[1], k1;
    int l1 = 1;
    int iw = 0;
    assert(in != out && work1 != work2);
    for (k1 = 2; k1 <= nf + 1; k1++) {
        int ip = ifac[k1];
        int l2 = ip * l1;
        int ido = n / l2;
        int idot = ido + ido;
#if PFFFT_TERMINAL_RADIX8
        if (ip == 6) {
            passf6_ps(idot, l1, in, out, isign);
        } else
#endif
            switch (ip) {
                case 8: {
                    passf8_ps(idot, l1, in, out, isign);
                } break;
                case 5: {
                    int ix2 = iw + idot;
                    int ix3 = ix2 + idot;
                    int ix4 = ix3 + idot;
                    passf5_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
                } break;
                case 4: {
                    int ix2 = iw + idot;
                    int ix3 = ix2 + idot;
                    passf4_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], isign);
                } break;
                case 2: {
                    passf2_ps(idot, l1, in, out, &wa[iw], isign);
                } break;
                case 3: {
                    int ix2 = iw + idot;
                    passf3_ps(idot, l1, in, out, &wa[iw], &wa[ix2], isign);
                } break;
                default: assert(0);
            }
        l1 = l2;
        iw += (ip - 1) * idot;
        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }

    return in; /* this is in fact the output .. */
}
#endif
#endif /* !PF_INPLACE_CODELETS */

#ifdef PF_FUSED_LAYOUT
#if !PF_INPLACE_CODELETS

/*
  Complex transforms deinterleave their input before the first pass and
  interleave the result of the last backward pass. The variants below do
  that while loading (first pass, l1 == 1) or storing (last pass, ido == 2):
  the element pair cc[i], cc[i+1] becomes PF_UNINTERLEAVE2(in[i], in[i+1])
  and ch[j], ch[j+1] becomes PF_INTERLEAVE2(..., out[j], out[j+1]). The
  arithmetic is that of passf2/3/4/5_ps, so the results are bit-identical
  and one full pass over the data disappears. The including file defines
  PF_FUSED_LAYOUT, PF_UNINTERLEAVE2 and PF_INTERLEAVE2.
*/

/* PF_(UN)INTERLEAVE2 with the outputs in p[i1], p[i2]; a backend whose VST
   is not a plain assignment defines these too */
#ifndef PF_INTERLEAVE2_ST
#define PF_INTERLEAVE2_ST(in1, in2, p, i1, i2)   PF_INTERLEAVE2(in1, in2, (p)[i1], (p)[i2])
#define PF_UNINTERLEAVE2_ST(in1, in2, p, i1, i2) PF_UNINTERLEAVE2(in1, in2, (p)[i1], (p)[i2])
#endif
#endif /* !PF_INPLACE_CODELETS */

static ALWAYS_INLINE(void) PF_CODELET_NAME(passf2_il)(int ido, const v4sf* PF_CODELET_RESTRICT in,
                                                      v4sf* PF_CODELET_RESTRICT ch, const float* wa1, float fsign) {
    int i;
    v4sf c0r, c0i, c1r, c1i;
    if (ido <= 2) {
        PF_UNINTERLEAVE2(VLD(in, 0), VLD(in, 1), c0r, c0i);
        PF_UNINTERLEAVE2(VLD(in, ido), VLD(in, ido + 1), c1r, c1i);
        VST(ch, 0, VADD(c0r, c1r));
        VST(ch, ido, VSUB(c0r, c1r));
        VST(ch, 1, VADD(c0i, c1i));
        VST(ch, ido + 1, VSUB(c0i, c1i));
    } else {
        for (i = 0; i < ido - 1; i += 2) {
            v4sf tr2, ti2;
            float wr, wi;
            PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i + 1), c0r, c0i);
            PF_UNINTERLEAVE2(VLD(in, i + ido), VLD(in, i + ido + 1), c1r, c1i);
            tr2 = VSUB(c0r, c1r);
            ti2 = VSUB(c0i, c1i);
            wr = wa1[i];
            wi = wa1[i + 1];
            VST(ch, i, VADD(c0r, c1r));
            VST(ch, i + 1, VADD(c0i, c1i));
            VCPLXMUL_FSIGN(tr2, ti2, wr, wi, fsign);
            VST(ch, i + ido, tr2);
            VST(ch, i + ido + 1, ti2);
        }
    }
}

static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf3_il)(int ido, const v4sf* PF_CODELET_RESTRICT in, v4sf* PF_CODELET_RESTRICT ch,
                               const float* wa1, const float* wa2, float fsign) {
    static const float taur = (float)-0.5;              /* cos(2*pi/3)   */
    float taui = (float)0.86602540378443864676 * fsign; /* sin(2*pi/3) = sqrt(3)/2 */
    int i;
    v4sf tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
    v4sf c0r, c0i, c1r, c1i, c2r, c2i;
    float wr1, wi1, wr2, wi2;
    assert(ido >= 2);
    for (i = 0; i < ido - 1; i += 2) {
        PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i + 1), c0r, c0i);
        PF_UNINTERLEAVE2(VLD(in, i + ido), VLD(in, i + ido + 1), c1r, c1i);
        PF_UNINTERLEAVE2(VLD(in, i + 2 * ido), VLD(in, i + 2 * ido + 1), c2r, c2i);
        tr2 = VADD(c1r, c2r);
        cr2 = SVMADD(taur, tr2, c0r);
        VST(ch, i, VADD(c0r, tr2));
        ti2 = VADD(c1i, c2i);
        ci2 = SVMADD(taur, ti2, c0i);
        VST(ch, i + 1, VADD(c0i, ti2));
        cr3 = SVMUL(taui, VSUB(c1r, c2r));
        ci3 = SVMUL(taui, VSUB(c1i, c2i));
        dr2 = VSUB(cr2, ci3);
        dr3 = VADD(cr2, ci3);
        di2 = VADD(ci2, cr3);
        di3 = VSUB(ci2, cr3);
        wr1 = wa1[i];
        wi1 = wa1[i + 1];
        wr2 = wa2[i];
        wi2 = wa2[i + 1];
        VCPLXMUL_FSIGN(dr2, di2, wr1, wi1, fsign);
        VST(ch, i + ido, dr2);
        VST(ch, i + ido + 1, di2);
        VCPLXMUL_FSIGN(dr3, di3, wr2, wi2, fsign);
        VST(ch, i + 2 * ido, dr3);
        VST(ch, i + 2 * ido + 1, di3);
    }
}

#if !PF_INPLACE_CODELETS
/* radix-4 butterfly of passf4_ps on one element pair of each leg; writes
   the four results to (y[2m], y[2m+1]) */
static ALWAYS_INLINE(void) passf4_il_butterfly(v4sf c0r, v4sf c0i, v4sf c1r, v4sf c1i, v4sf c2r, v4sf c2i, v4sf c3r,
                                               v4sf c3i, float fsign, v4sf* y) {
    v4sf tr1 = VSUB(c0r, c2r);
    v4sf tr2 = VADD(c0r, c2r);
    v4sf ti1 = VSUB(c0i, c2i);
    v4sf ti2 = VADD(c0i, c2i);
    v4sf ti4 = VSUB(c1r, c3r);
    v4sf tr4 = VSUB(c3i, c1i);
    v4sf tr3 = VADD(c1r, c3r);
    v4sf ti3 = VADD(c1i, c3i);
    y[0] = VADD(tr2, tr3);
    y[1] = VADD(ti2, ti3);
    y[2] = VADD_FSIGN(tr1, tr4, fsign);
    y[3] = VADD_FSIGN(ti1, ti4, fsign);
    y[4] = VSUB(tr2, tr3);
    y[5] = VSUB(ti2, ti3);
    y[6] = VSUB_FSIGN(tr1, tr4, fsign);
    y[7] = VSUB_FSIGN(ti1, ti4, fsign);
}
#endif /* !PF_INPLACE_CODELETS */

static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf4_il)(int ido, const v4sf* PF_CODELET_RESTRICT in, v4sf* PF_CODELET_RESTRICT ch,
                               const float* wa1, const float* wa2, const float* wa3, float fsign) {
    int i;
    v4sf c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i;
    if (ido == 2) {
        v4sf y[8];
        PF_UNINTERLEAVE2(VLD(in, 0), VLD(in, 1), c0r, c0i);
        PF_UNINTERLEAVE2(VLD(in, ido), VLD(in, ido + 1), c1r, c1i);
        PF_UNINTERLEAVE2(VLD(in, 2 * ido), VLD(in, 2 * ido + 1), c2r, c2i);
        PF_UNINTERLEAVE2(VLD(in, 3 * ido), VLD(in, 3 * ido + 1), c3r, c3i);
        passf4_il_butterfly(c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, fsign, y);
        VST(ch, 0 * ido + 0, y[0]);
        VST(ch, 0 * ido + 1, y[1]);
        VST(ch, 1 * ido + 0, y[2]);
        VST(ch, 1 * ido + 1, y[3]);
        VST(ch, 2 * ido + 0, y[4]);
        VST(ch, 2 * ido + 1, y[5]);
        VST(ch, 3 * ido + 0, y[6]);
        VST(ch, 3 * ido + 1, y[7]);
    } else {
        for (i = 0; i < ido - 1; i += 2) {
            v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
            float wr1, wi1, wr2, wi2, wr3, wi3;
            PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i + 1), c0r, c0i);
            PF_UNINTERLEAVE2(VLD(in, i + ido), VLD(in, i + ido + 1), c1r, c1i);
            PF_UNINTERLEAVE2(VLD(in, i + 2 * ido), VLD(in, i + 2 * ido + 1), c2r, c2i);
            PF_UNINTERLEAVE2(VLD(in, i + 3 * ido), VLD(in, i + 3 * ido + 1), c3r, c3i);
            tr1 = VSUB(c0r, c2r);
            tr2 = VADD(c0r, c2r);
            ti1 = VSUB(c0i, c2i);
            ti2 = VADD(c0i, c2i);
            tr4 = VSUB(c3i, c1i);
            ti4 = VSUB(c1r, c3r);
            tr3 = VADD(c1r, c3r);
            ti3 = VADD(c1i, c3i);

            VST(ch, i, VADD(tr2, tr3));
            cr3 = VSUB(tr2, tr3);
            VST(ch, i + 1, VADD(ti2, ti3));
            ci3 = VSUB(ti2, ti3);

            cr2 = VADD_FSIGN(tr1, tr4, fsign);
            cr4 = VSUB_FSIGN(tr1, tr4, fsign);
            ci2 = VADD_FSIGN(ti1, ti4, fsign);
            ci4 = VSUB_FSIGN(ti1, ti4, fsign);
            wr1 = wa1[i];
            wi1 = wa1[i + 1];
            VCPLXMUL_FSIGN(cr2, ci2, wr1, wi1, fsign);
            wr2 = wa2[i];
            wi2 = wa2[i + 1];
            VST(ch, i + ido, cr2);
            VST(ch, i + ido + 1, ci2);

            VCPLXMUL_FSIGN(cr3, ci3, wr2, wi2, fsign);
            wr3 = wa3[i];
            wi3 = wa3[i + 1];
            VST(ch, i + 2 * ido, cr3);
            VST(ch, i + 2 * ido + 1, ci3);

            VCPLXMUL_FSIGN(cr4, ci4, wr3, wi3, fsign);
            VST(ch, i + 3 * ido, cr4);
            VST(ch, i + 3 * ido + 1, ci4);
        }
    }
}

static ALWAYS_INLINE(void)
    PF_CODELET_NAME(passf5_il)(int ido, const v4sf* PF_CODELET_RESTRICT in, v4sf* PF_CODELET_RESTRICT ch,
                               const float* wa1, const float* wa2, const float* wa3, const float* wa4, float fsign) {
    static const float tr11 = (float)0.30901699437494742410;  /* cos(2*pi/5) = (sqrt(5)-1)/4 */
    const float ti11 = (float)0.95105651629515357212 * fsign; /* sin(2*pi/5) */
    static const float tr12 = (float)-0.80901699437494742410; /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
    const float ti12 = (float)0.58778525229247312917 * fsign; /* sin(4*pi/5) */
    int i, j;
    v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3,
        tr4, tr5;
    float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

    assert(ido >= 2);
    for (i = 0; i < ido - 1; i += 2) {
        v4sf xr[6], xi[6]; /* legs 1..5 as in passf5_ps */
        for (j = 1; j <= 5; ++j) {
            PF_UNINTERLEAVE2(VLD(in, (j - 1) * ido + i), VLD(in, (j - 1) * ido + i + 1), xr[j], xi[j]);
        }
        ti5 = VSUB(xi[2], xi[5]);
        ti2 = VADD(xi[2], xi[5]);
        ti4 = VSUB(xi[3], xi[4]);
        ti3 = VADD(xi[3], xi[4]);
        tr5 = VSUB(xr[2], xr[5]);
        tr2 = VADD(xr[2], xr[5]);
        tr4 = VSUB(xr[3], xr[4]);
        tr3 = VADD(xr[3], xr[4]);
        VST(ch, i, VADD(xr[1], VADD(tr2, tr3)));
        VST(ch, i + 1, VADD(xi[1], VADD(ti2, ti3)));
        cr2 = VADD(xr[1], SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
        ci2 = VADD(xi[1], SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
        cr3 = VADD(xr[1], SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
        ci3 = VADD(xi[1], SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
        cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
        ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
        cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
        ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
        dr3 = VSUB(cr3, ci4);
        dr4 = VADD(cr3, ci4);
        di3 = VADD(ci3, cr4);
        di4 = VSUB(ci3, cr4);
        dr5 = VADD(cr2, ci5);
        dr2 = VSUB(cr2, ci5);
        di5 = VSUB(ci2, cr5);
        di2 = VADD(ci2, cr5);
        wr1 = wa1[i];
        wi1 = wa1[i + 1];
        wr2 = wa2[i];
        wi2 = wa2[i + 1];
        wr3 = wa3[i];
        wi3 = wa3[i + 1];
        wr4 = wa4[i];
        wi4 = wa4[i + 1];
        VCPLXMUL_FSIGN(dr2, di2, wr1, wi1, fsign);
        VST(ch, ido + i, dr2);
        VST(ch, ido + i + 1, di2);
        VCPLXMUL_FSIGN(dr3, di3, wr2, wi2, fsign);
        VST(ch, 2 * ido + i, dr3);
        VST(ch, 2 * ido + i + 1, di3);
        VCPLXMUL_FSIGN(dr4, di4, wr3, wi3, fsign);
        VST(ch, 3 * ido + i, dr4);
        VST(ch, 3 * ido + i + 1, di4);
        VCPLXMUL_FSIGN(dr5, di5, wr4, wi4, fsign);
        VST(ch, 4 * ido + i, dr5);
        VST(ch, 4 * ido + i + 1, di5);
    }
}

#if !PF_INPLACE_CODELETS
/* the four deinterleaving codelets as functions, for either direction */
static NEVER_INLINE(void)
    passf2_il_ps(int ido, const v4sf* RESTRICT in, v4sf* RESTRICT ch, const float* wa1, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf2_il_op_body, ido, in, ch, wa1);
}

static NEVER_INLINE(void)
    passf3_il_ps(int ido, const v4sf* RESTRICT in, v4sf* RESTRICT ch, const float* wa1, const float* wa2, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf3_il_op_body, ido, in, ch, wa1, wa2);
}

static NEVER_INLINE(void) passf4_il_ps(int ido, const v4sf* RESTRICT in, v4sf* RESTRICT ch, const float* wa1,
                                       const float* wa2, const float* wa3, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf4_il_op_body, ido, in, ch, wa1, wa2, wa3);
}

static NEVER_INLINE(void) passf5_il_ps(int ido, const v4sf* RESTRICT in, v4sf* RESTRICT ch, const float* wa1,
                                       const float* wa2, const float* wa3, const float* wa4, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf5_il_op_body, ido, in, ch, wa1, wa2, wa3, wa4);
}

/* last pass (ido == 2: one complex element per leg) of passf4_ps, storing
   interleaved complex output */
static ALWAYS_INLINE(void) passf4_ol_body(int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT out, float fsign) {
    const int ido = 2, l1ido = l1 * ido;
    int k;
    for (k = 0; k < l1ido; k += ido, out += ido, cc += 4 * ido) {
        v4sf y[8];
        passf4_il_butterfly(VLD(cc, 0), VLD(cc, 1), VLD(cc, ido + 0), VLD(cc, ido + 1), VLD(cc, 2 * ido + 0),
                            VLD(cc, 2 * ido + 1), VLD(cc, 3 * ido + 0), VLD(cc, 3 * ido + 1), fsign, y);
        PF_INTERLEAVE2_ST(y[0], y[1], out, 0 * l1ido + 0, 0 * l1ido + 1);
        PF_INTERLEAVE2_ST(y[2], y[3], out, 1 * l1ido + 0, 1 * l1ido + 1);
        PF_INTERLEAVE2_ST(y[4], y[5], out, 2 * l1ido + 0, 2 * l1ido + 1);
        PF_INTERLEAVE2_ST(y[6], y[7], out, 3 * l1ido + 0, 3 * l1ido + 1);
    }
}

static NEVER_INLINE(void) passf4_ol_ps(int l1, const v4sf* RESTRICT cc, v4sf* RESTRICT out, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf4_ol_body, l1, cc, out);
}

#if PFFFT_TERMINAL_RADIX8
/* passf8_ps (ido == 2: the last pass), storing interleaved complex output;
   the same arithmetic, so the result has the bits of passf8_ps followed by
   PF_INTERLEAVE2 */
static NEVER_INLINE(void) passf8_ol_ps(int l1, const v4sf* cc, v4sf* out, float fsign) {
    const float c8 = (float)M_SQRT1_2; /* 1/sqrt(2) */
    const v4sf k8 = LD_PS1(c8);        /* named: see passf8_ps */
    const int ido = 2, l1ido = l1 * ido;
    int k;
    if (fsign < 0) {
#define PF_ST8(m, a, b) PF_INTERLEAVE2_ST(a, b, out, (m) * l1ido, (m) * l1ido + 1);
        for (k = 0; k < l1ido; k += ido, out += ido, cc += 8 * ido) {
            PF_PASS8(cc, 0, 1, PF_ST8)
        }
#undef PF_ST8
    } else {
#define PF_ST8(m, a, b) PF_INTERLEAVE2_ST(b, a, out, (m) * l1ido, (m) * l1ido + 1);
        for (k = 0; k < l1ido; k += ido, out += ido, cc += 8 * ido) {
            PF_PASS8(cc, 1, 0, PF_ST8)
        }
#undef PF_ST8
    }
}

/* passf6_ps, storing interleaved complex output, as passf8_ol_ps */
static NEVER_INLINE(void) passf6_ol_ps(int l1, const v4sf* cc, v4sf* out, float fsign) {
    const float ctaur = -0.5f, ctaui = (float)0.86602540378443864676; /* cos, sin(2*pi/3) */
    const v4sf taur = LD_PS1(ctaur), taui = LD_PS1(ctaui);            /* named: see passf8_ps */
    const int ido = 2, l1ido = l1 * ido;
    int k;
    if (fsign < 0) {
#define PF_ST6(m, a, b) PF_INTERLEAVE2_ST(a, b, out, (m) * l1ido, (m) * l1ido + 1);
        for (k = 0; k < l1ido; k += ido, out += ido, cc += 6 * ido) {
            PF_PASS6(cc, 0, 1, PF_ST6)
        }
#undef PF_ST6
    } else {
#define PF_ST6(m, a, b) PF_INTERLEAVE2_ST(b, a, out, (m) * l1ido, (m) * l1ido + 1);
        for (k = 0; k < l1ido; k += ido, out += ido, cc += 6 * ido) {
            PF_PASS6(cc, 1, 0, PF_ST6)
        }
#undef PF_ST6
    }
}
#endif /* PFFFT_TERMINAL_RADIX8 */

#if PFFFT_INPLACE_FIRST_PASS || PFFFT_DEPTH_FIRST
/*
  The first complex pass in place. In the first pass (l1 == 1) each
  butterfly reads the element pair at position i of every leg and writes
  its results to the same positions, so the pass can run with input and
  output the same buffer, provided that no butterfly reads a position after
  storing to it. The drivers below use that where a fused (de)interleaving
  pass would otherwise have to write into its own input: running the first
  pass in place exchanges the buffers of the later passes, and the separate
  (de)interleaving sweep over the data is not needed.

  The bodies meet the condition as follows. passf4_ps (ido > 2) and the
  four deinterleaving codelets passf2/3/4/5_il_ps load every leg before
  their first store. passf2_ps (ido > 2) and passf3_ps store the outputs of
  leg 0 between their loads, but each position is read only before the
  store to it. passf5_ps reads leg 1 again after storing to it; its in-place
  instantiation loads leg 1 into two locals before the first store
  (cc_leg1r, cc_leg1i). The plain radix-2 codelet with one element per leg
  (ido <= 2) stores ch[0] and then reads cc[0] again, so it must not run in
  place: passf2_ip_ps asserts ido > 2, and the driver takes the first pass
  in place only when another pass follows, which for a radix-2 first pass
  means at least two elements per leg. (Its deinterleaving form,
  passf2_il_ps, loads both legs first and is safe at any ido.)

  The bodies come from a second inclusion of this file, with
  PF_INPLACE_CODELETS 1, which compiles only the eight bodies once more
  under the names passfR_body and passfR_il_body, without 'restrict'. The
  regular codelets keep 'restrict' and their code. As for every inclusion
  of this file, the includer (here the file itself) sets the parameter
  before and resets it after; the rest of this inclusion continues with 0.
*/
#undef PF_INPLACE_CODELETS
#define PF_INPLACE_CODELETS 1
#include "pffft_passes_impl.h"
#undef PF_INPLACE_CODELETS
#define PF_INPLACE_CODELETS 0

/* the depth-first passes use the radix-4 one only, and take their input
   from another buffer */
static NEVER_INLINE(void)
    passf4_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, const float* wa3, float fsign) {
    assert(ido > 2);
    PF_PASS_DIRECTIONS(fsign, passf4_body, ido, 1, c, c, wa1, wa2, wa3);
}

#if PFFFT_INPLACE_FIRST_PASS
static NEVER_INLINE(void) passf2_ip_ps(int ido, v4sf* c, const float* wa1, float fsign) {
    assert(ido > 2); /* see above: one element per leg is not safe in place */
    PF_PASS_DIRECTIONS(fsign, passf2_body, ido, 1, c, c, wa1);
}

static NEVER_INLINE(void) passf3_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, float fsign) {
    assert(ido > 2);
    PF_PASS_DIRECTIONS(fsign, passf3_body, ido, 1, c, c, wa1, wa2);
}

static NEVER_INLINE(void) passf5_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, const float* wa3,
                                       const float* wa4, float fsign) {
    assert(ido > 2);
    PF_PASS_DIRECTIONS(fsign, passf5_body, ido, 1, c, c, wa1, wa2, wa3, wa4);
}

static NEVER_INLINE(void) passf2_il_ip_ps(int ido, v4sf* c, const float* wa1, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf2_il_body, ido, c, c, wa1);
}

static NEVER_INLINE(void) passf3_il_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf3_il_body, ido, c, c, wa1, wa2);
}

static NEVER_INLINE(void)
    passf4_il_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, const float* wa3, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf4_il_body, ido, c, c, wa1, wa2, wa3);
}

static NEVER_INLINE(void) passf5_il_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, const float* wa3,
                                          const float* wa4, float fsign) {
    PF_PASS_DIRECTIONS(fsign, passf5_il_body, ido, c, c, wa1, wa2, wa3, wa4);
}
#endif /* PFFFT_INPLACE_FIRST_PASS */
#endif /* PFFFT_INPLACE_FIRST_PASS || PFFFT_DEPTH_FIRST */

/* the passes of cfftf1_ps from factor index k1 on, l1 and iw as cfftf1_ps
   has them there; 'in' holds the data and 'out' is the other buffer. With
   'interleave' set, the last pass stores interleaved into il_out when it
   can (radix 4, or radix 8 or 6 with PFFFT_TERMINAL_RADIX8, not reading
   il_out); otherwise the result is interleaved into il_out afterwards, in
   a separate sweep. With PFFFT_INPLACE_FIRST_PASS, when the last pass
   would read il_out and the first pass (l1 == 1, not radix 8 or 6) is
   followed by another, the first pass runs in place in 'in' (passfR_ip_ps
   above), so that the last pass reads the other buffer. Returns the
   buffer holding the result. A flag rather than a NULL il_out selects
   this: callers pass one of their two buffers as il_out, and testing it
   for NULL would tell a static analyzer that that buffer, too, may be
   NULL. Callers use the wrappers below, which pass the flag as a
   constant. */
static ALWAYS_INLINE(v4sf*) cfftf1_passes_ps(int n, v4sf* in, v4sf* out, int k1, int l1, int iw, const float* wa,
                                             const int* ifac, int isign, int interleave, v4sf* il_out) {
    const int nf = ifac[1];
    assert(in != out);
#if PFFFT_INPLACE_FIRST_PASS
    /* the passes k1 .. nf+1 alternate between in and out, so the last one
     reads 'in' when their number is odd. An if-chain, not a switch: no
     jump table in the 8-lane code (check_isa_baseline) */
    if (interleave && l1 == 1 && k1 <= nf && ifac[k1] != 8 && ifac[k1] != 6
        && (((nf + 2 - k1) & 1) ? in : out) == il_out) {
        const int ip = ifac[k1], idot = 2 * (n / ip);
        const float fsign = (float)isign;
        if (ip == 4) {
            passf4_ip_ps(idot, in, &wa[iw], &wa[iw + idot], &wa[iw + 2 * idot], fsign);
        } else if (ip == 2) {
            passf2_ip_ps(idot, in, &wa[iw], fsign);
        } else if (ip == 3) {
            passf3_ip_ps(idot, in, &wa[iw], &wa[iw + idot], fsign);
        } else {
            assert(ip == 5);
            passf5_ip_ps(idot, in, &wa[iw], &wa[iw + idot], &wa[iw + 2 * idot], &wa[iw + 3 * idot], fsign);
        }
        l1 = ip;
        iw += (ip - 1) * idot;
        ++k1;
    }
#endif
    for (; k1 <= nf + 1; k1++) {
        int ip = ifac[k1];
        int l2 = ip * l1;
        int ido = n / l2;
        int idot = ido + ido;
        v4sf* tmp;
        if (interleave && k1 == nf + 1 && ip == 4 && in != il_out) {
            passf4_ol_ps(l1, in, il_out, (float)isign);
            return il_out;
        }
#if PFFFT_TERMINAL_RADIX8
        if (interleave && k1 == nf + 1 && ip == 8 && in != il_out) {
            passf8_ol_ps(l1, in, il_out, (float)isign);
            return il_out;
        }
        if (interleave && k1 == nf + 1 && ip == 6 && in != il_out) {
            passf6_ol_ps(l1, in, il_out, (float)isign);
            return il_out;
        }
        /* in front of the switch, as in rfftf1_ps */
        if (ip == 6) {
            passf6_ps(idot, l1, in, out, isign);
        } else
#endif
            switch (ip) {
                case 8: {
                    passf8_ps(idot, l1, in, out, isign);
                } break;
                case 5: {
                    int ix2 = iw + idot;
                    int ix3 = ix2 + idot;
                    int ix4 = ix3 + idot;
                    passf5_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
                } break;
                case 4: {
                    int ix2 = iw + idot;
                    int ix3 = ix2 + idot;
                    passf4_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], isign);
                } break;
                case 2: {
                    passf2_ps(idot, l1, in, out, &wa[iw], isign);
                } break;
                case 3: {
                    int ix2 = iw + idot;
                    passf3_ps(idot, l1, in, out, &wa[iw], &wa[ix2], isign);
                } break;
                default: assert(0);
            }
        l1 = l2;
        iw += (ip - 1) * idot;
        tmp = in;
        in = out;
        out = tmp;
    }
    if (interleave) {
        int k;
        for (k = 0; k < n; ++k) {
            PF_INTERLEAVE2_ST(VLD(in, 2 * k), VLD(in, 2 * k + 1), il_out, 2 * k, 2 * k + 1);
        }
        return il_out;
    }
    return in;
}

/* cfftf1_passes_ps without interleaving: returns the buffer holding the
   result, in or out */
static v4sf*
cfftf1_tail_ps(int n, v4sf* in, v4sf* out, int k1, int l1, int iw, const float* wa, const int* ifac, int isign) {
    return cfftf1_passes_ps(n, in, out, k1, l1, iw, wa, ifac, isign, 0, NULL);
}

/* cfftf1_passes_ps with the result interleaved into il_out, which may be
   in or out */
static void
cfftf1_tail_ol_ps(int n, v4sf* in, v4sf* out, int k1, int l1, int iw, const float* wa, const int* ifac, int isign,
                  v4sf* il_out) {
    cfftf1_passes_ps(n, in, out, k1, l1, iw, wa, ifac, isign, 1, il_out);
}

/* cfftf1_ps(n, work2, work1, work2, ...) for data that is still interleaved
   in 'input': the first pass deinterleaves it while loading and writes work1.
   input may be work2. If it is work1, the first pass deinterleaves in place
   in work1 with PFFFT_INPLACE_FIRST_PASS (passfR_il_ip_ps above); without
   it, and for the single radix-8 or (with PFFFT_TERMINAL_RADIX8) radix-6
   pass of n == 8 or 6, the deinterleaving is a separate pass into work2,
   as without fusion.
   Returns the buffer holding the result. */
static v4sf*
cfftf1_il_ps(int n, const v4sf* input, v4sf* work1, v4sf* work2, const float* wa, const int* ifac, int isign) {
    const int ip = ifac[2];
    const int ido = n / ip, idot = ido + ido;
    const float fsign = (float)isign;
    int k;
    assert(work1 != work2);
#if PFFFT_INPLACE_FIRST_PASS
    if (input == work1 && ip != 8 && ip != 6) {
        /* an if-chain, not a switch: no jump table in the 8-lane code */
        if (ip == 4) {
            passf4_il_ip_ps(idot, work1, &wa[0], &wa[idot], &wa[2 * idot], fsign);
        } else if (ip == 2) {
            passf2_il_ip_ps(idot, work1, &wa[0], fsign);
        } else if (ip == 3) {
            passf3_il_ip_ps(idot, work1, &wa[0], &wa[idot], fsign);
        } else {
            assert(ip == 5);
            passf5_il_ip_ps(idot, work1, &wa[0], &wa[idot], &wa[2 * idot], &wa[3 * idot], fsign);
        }
        return cfftf1_tail_ps(n, work1, work2, 3, ip, (ip - 1) * idot, wa, ifac, isign);
    }
#endif
    if (input != work1 && ip != 8 && ip != 6) {
        switch (ip) {
            case 5: passf5_il_ps(idot, input, work1, &wa[0], &wa[idot], &wa[2 * idot], &wa[3 * idot], fsign); break;
            case 4: passf4_il_ps(idot, input, work1, &wa[0], &wa[idot], &wa[2 * idot], fsign); break;
            case 3: passf3_il_ps(idot, input, work1, &wa[0], &wa[idot], fsign); break;
            default:
                assert(ip == 2);
                passf2_il_ps(idot, input, work1, &wa[0], fsign);
                break;
        }
        return cfftf1_tail_ps(n, work1, work2, 3, ip, (ip - 1) * idot, wa, ifac, isign);
    }
    for (k = 0; k < n; ++k) {
        PF_UNINTERLEAVE2_ST(VLD(input, 2 * k), VLD(input, 2 * k + 1), work2, 2 * k, 2 * k + 1);
    }
    return cfftf1_tail_ps(n, work2, work1, 2, 1, 0, wa, ifac, isign);
}

#if PFFFT_DEPTH_FIRST
/*
  Depth-first complex transforms.

  The passes above sweep the whole buffer once each, and every pass writes
  the buffer that the next one reads: beyond the size of a cache level, each
  pass moves all the data through that level again. The functions below run
  the leading radix-4 passes of a plan in another order.

  forward   A radix-4 pass over a group of 4*ido elements leaves four
            independent sub-transforms in its four legs. In place (the
            codelets of the in-place first pass above), leg j stays where
            it was, so the pass can be followed by all the remaining passes
            of leg 0, then by those of leg 1, and so on: depth first. A
            sub-transform is finished while it is in the cache. The last
            passes of a sub-transform, its leaf, are the passes above,
            between the leaf and scratch of its size; they leave the leaf
            in natural order.
  backward  The same tree from the leaves up: the leaves first, then the
            radix-4 passes as decimation-in-time passes (passb4_dit_*
            below), which multiply by the twiddles before the butterfly
            where the passes above do it after; the last one interleaves
            the result. These are other operations than the passes above
            perform: the results differ in the last bits. Forward, the
            operations and their operands are the same, and so are the
            results.

  Bins. A forward in-place pass leaves in leg j the bins that are j modulo
  4. After 'levels' of them a buffer of n elements holds 4^levels regions
  of n / 4^levels elements each: region
  J = c1 4^(levels-1) + c2 4^(levels-2) + ... + c_levels holds, in natural
  order of q, the bins c1 + 4 c2 + ... + 4^(levels-1) c_levels + 4^levels q:
  J is that residue with its base-4 digits reversed. The backward passes
  take the bins in the same places.

  The spectrum keeps its layout, in two steps:

  streams   The steps that combine the lanes (finalize, preprocess) work
            on blocks of as many consecutive bins as there are lanes. After
            'top' passes, 1 with 4 lanes and 2 with 8, these bins are one
            element of each of 4 (8) regions, the same element of each: a
            block is read from, or written to, streams that advance
            together, with the regions, called slices here, in natural
            order. Finalize and preprocess have a variant for that.
  slices    A slice larger than a leaf continues depth first with 'deep'
            more passes, in scratch, and is brought back into natural
            order when it is done (cfftf1_df_natural_ps): the elements of its
            regions, each a leaf in natural order, are interleaved, while
            they are still in the cache for all but the largest
            transforms. This is the one step the other schedule does not
            have; it moves the data once more, without arithmetic.
*/

/* Whether the plan in ifac of a transform runs its first 'top' passes
   depth first (top: 1 with 4 lanes, 2 with 8, see above): they have to be
   radix-4 passes, and two more passes make the smallest leaf. */
static int
cfft_df_top(const int* ifac, int top) {
    int k;
    if (ifac[1] < top + 2) {
        return 0;
    }
    for (k = 0; k < top; ++k) {
        if (ifac[2 + k] != 4) {
            return 0;
        }
    }
    return 1;
}

/* The number of passes after the first 'top' that run depth first within
   a slice of n elements: radix-4 passes, as many as it takes to make a
   leaf of at most leaf_bytes bytes while two passes remain for the leaf.
   0: the slice is a leaf. */
static int
cfft_df_deep(const int* ifac, int n, int top, size_t leaf_bytes) {
    const size_t leaf_max = leaf_bytes / (2 * sizeof(v4sf));
    const int nf = ifac[1];
    int deep = 0;
    while (top + deep < nf - 2 && ifac[2 + top + deep] == 4 && (size_t)n > leaf_max) {
        ++deep;
        n /= 4;
    }
    return deep;
}

/* rev[J] for J < regions, a power of 4: J with its base-4 digits reversed */
static void
cfft_df_rev(int* rev, int regions) {
    int J;
    for (J = 0; J < regions; ++J) {
        int r = 0, v = J, m;
        for (m = regions; m > 1; m /= 4) {
            r = 4 * r + v % 4;
            v /= 4;
        }
        rev[J] = r;
    }
}

/*
  passb4_dit: the radix-4 decimation-in-time pass of the backward transform
  over one group (l1 == 1): legs 1, 2, 3 times their twiddles, then the
  butterfly of passf4 with fsign = +1. Every leg is loaded before the first
  store and position i of the legs is all a butterfly reads and writes, so
  cc and ch may be the same buffer. With 'interleave' the result is stored
  as interleaved complex data.
*/
static ALWAYS_INLINE(void) passb4_dit_body(int ido, const v4sf* cc, v4sf* ch, const float* wa1, const float* wa2,
                                           const float* wa3, const int interleave) {
    int i;
    for (i = 0; i < ido - 1; i += 2) {
        v4sf c0r = VLD(cc, i), c0i = VLD(cc, i + 1);
        v4sf c1r = VLD(cc, i + ido), c1i = VLD(cc, i + ido + 1);
        v4sf c2r = VLD(cc, i + 2 * ido), c2i = VLD(cc, i + 2 * ido + 1);
        v4sf c3r = VLD(cc, i + 3 * ido), c3i = VLD(cc, i + 3 * ido + 1);
        v4sf tr1, tr2, tr3, tr4, ti1, ti2, ti3, ti4;
        v4sf y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        VCPLXMUL(c1r, c1i, LD_PS1(wa1[i]), LD_PS1(wa1[i + 1]));
        VCPLXMUL(c2r, c2i, LD_PS1(wa2[i]), LD_PS1(wa2[i + 1]));
        VCPLXMUL(c3r, c3i, LD_PS1(wa3[i]), LD_PS1(wa3[i + 1]));
        tr1 = VSUB(c0r, c2r);
        tr2 = VADD(c0r, c2r);
        ti1 = VSUB(c0i, c2i);
        ti2 = VADD(c0i, c2i);
        ti4 = VSUB(c1r, c3r);
        tr4 = VSUB(c3i, c1i);
        tr3 = VADD(c1r, c3r);
        ti3 = VADD(c1i, c3i);
        y0r = VADD(tr2, tr3);
        y0i = VADD(ti2, ti3);
        y1r = VADD(tr1, tr4);
        y1i = VADD(ti1, ti4);
        y2r = VSUB(tr2, tr3);
        y2i = VSUB(ti2, ti3);
        y3r = VSUB(tr1, tr4);
        y3i = VSUB(ti1, ti4);
        if (interleave) {
            PF_INTERLEAVE2_ST(y0r, y0i, ch, i, i + 1);
            PF_INTERLEAVE2_ST(y1r, y1i, ch, i + ido, i + ido + 1);
            PF_INTERLEAVE2_ST(y2r, y2i, ch, i + 2 * ido, i + 2 * ido + 1);
            PF_INTERLEAVE2_ST(y3r, y3i, ch, i + 3 * ido, i + 3 * ido + 1);
        } else {
            VST(ch, i, y0r);
            VST(ch, i + 1, y0i);
            VST(ch, i + ido, y1r);
            VST(ch, i + ido + 1, y1i);
            VST(ch, i + 2 * ido, y2r);
            VST(ch, i + 2 * ido + 1, y2i);
            VST(ch, i + 3 * ido, y3r);
            VST(ch, i + 3 * ido + 1, y3i);
        }
    }
}

/* in place */
static NEVER_INLINE(void) passb4_dit_ip_ps(int ido, v4sf* c, const float* wa1, const float* wa2, const float* wa3) {
    passb4_dit_body(ido, c, c, wa1, wa2, wa3, 0);
}

/* from cc to ch */
static NEVER_INLINE(void)
    passb4_dit_op_ps(int ido, const v4sf* cc, v4sf* ch, const float* wa1, const float* wa2, const float* wa3) {
    passb4_dit_body(ido, cc, ch, wa1, wa2, wa3, 0);
}

/* from cc to out, interleaved */
static NEVER_INLINE(void)
    passb4_dit_ol_ps(int ido, const v4sf* cc, v4sf* out, const float* wa1, const float* wa2, const float* wa3) {
    passb4_dit_body(ido, cc, out, wa1, wa2, wa3, 1);
}

/* A leaf: the passes from factor index k1 on of the sub-transform of n
   elements in x, iw as cfftf1_ps has it there, left in x in natural order.
   tmp is scratch for 2*n vectors. The passes alternate between x and tmp,
   so an odd number of them starts with one in place: a radix-4 pass, as a
   plan that starts with radix-4 passes has no other factor before its
   last one (cffti1_factors). */
static void
cfft_df_leaf_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa, const int* ifac, int isign) {
    const v4sf* result;
    if ((ifac[1] + 2 - k1) & 1) {
        const int idot = 2 * (n / 4);
        assert(ifac[k1] == 4);
        passf4_ip_ps(idot, x, &wa[iw], &wa[iw + idot], &wa[iw + 2 * idot], (float)isign);
        result = cfftf1_tail_ps(n, x, tmp, k1 + 1, 4, iw + 3 * idot, wa, ifac, isign);
    } else {
        result = cfftf1_tail_ps(n, x, tmp, k1, 1, iw, wa, ifac, isign);
    }
    assert(result == x);
    (void)result; /* otherwise unused when assertions are disabled */
}

/* Forward: the passes from factor index k1 on of the sub-transform of n
   elements in x, the next 'levels' of them radix-4 passes in place, then
   the leaves, each with tmp as its scratch: x is left with the bins of
   4^levels regions as described above. Depth first without recursion: the
   leaves in sequence, each after the passes of the groups that begin with
   it, the largest group first. */
static NEVER_INLINE(void)
    cfft_df_forward_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa, const int* ifac, int levels) {
    const int leaves = 1 << (2 * levels), nleaf = n / leaves;
    int leaf, level;
    for (leaf = 0; leaf < leaves; ++leaf, x += 2 * nleaf) {
        int idot = 2 * (n / 4), w = iw; /* a leg of the group of a level, and its twiddles */
        for (level = 0; level < levels; ++level, w += 3 * idot, idot /= 4) {
            if (leaf % (leaves >> (2 * level)) == 0) {
                passf4_ip_ps(idot, x, &wa[w], &wa[w + idot], &wa[w + 2 * idot], -1.f);
            }
        }
        cfft_df_leaf_ps(nleaf, x, tmp, k1 + levels, w, wa, ifac, -1);
    }
}

/* Backward: as cfft_df_forward_ps, x holding the bins of the regions: each
   leaf, then the passes of the groups that end with it, the smallest group
   first. */
static NEVER_INLINE(void)
    cfft_df_backward_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa, const int* ifac, int levels) {
    const int leaves = 1 << (2 * levels), nleaf = n / leaves;
    /* the twiddles of the leaves follow those of the levels, three legs of
       2 * n / 4^(level + 1) floats at each level: 2 * (n - nleaf) in all.
       (A loop that adds them up is a candidate for a compiler's vectorizer,
       and MSVC's emits instructions beyond the baseline for it, selected
       at run time, which check_isa_baseline rejects.) */
    const int w_leaf = iw + 2 * (n - nleaf);
    int leaf, level;
    for (leaf = 0; leaf < leaves; ++leaf) {
        int idot = 2 * nleaf, w = w_leaf; /* a leg of the group of a level, and its twiddles */
        cfft_df_leaf_ps(nleaf, x + 2 * leaf * nleaf, tmp, k1 + levels, w_leaf, wa, ifac, +1);
        for (level = levels - 1; level >= 0; --level, idot *= 4) {
            const int group = leaves >> (2 * level); /* its leaves */
            w -= 3 * idot;
            if ((leaf + 1) % group == 0) {
                passb4_dit_ip_ps(idot, x + 2 * (leaf + 1 - group) * nleaf, &wa[w], &wa[w + idot], &wa[w + 2 * idot]);
            }
        }
    }
}

/* Forward: the passes from factor index k1 on of the sub-transform of n
   elements in x, the first 'deep' of them depth first, left in x in
   natural order. deep > 0: the first pass moves x to tmp, where the rest
   runs; then region rev[c] of tmp, a leaf in natural order, holds the
   bins c + 4^deep q, which go back to x. rev: cfft_df_rev for 4^deep
   regions. tmp is scratch for 2 * n vectors and for a leaf's. */
static NEVER_INLINE(void) cfftf1_df_natural_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa,
                                               const int* ifac, int deep, const int* rev) {
    if (deep > 0) {
        const int idot = 2 * (n / 4), regions = 1 << (2 * deep), nleaf = n / regions;
        const int tile = regions < (PFFFT_DEPTH_FIRST_TILE) ? regions : (PFFFT_DEPTH_FIRST_TILE);
        v4sf* leaf_tmp = tmp + 2 * n;
        int j, q, c, c0;
        passf4_ps(idot, 1, x, tmp, &wa[iw], &wa[iw + idot], &wa[iw + 2 * idot], -1.f);
        for (j = 0; j < 4; ++j) {
            cfft_df_forward_ps(n / 4, tmp + j * idot, leaf_tmp, k1 + 1, iw + 3 * idot, wa, ifac, deep - 1);
        }
        /* a tile of regions at a time: see PFFFT_DEPTH_FIRST_TILE */
        for (c0 = 0; c0 < regions; c0 += tile) {
            for (q = 0; q < nleaf; ++q) {
                v4sf* to = x + 2 * (c0 + regions * q);
                for (c = c0; c < c0 + tile; ++c, to += 2) {
                    const v4sf* from = tmp + 2 * (rev[c] * nleaf + q);
                    VST(to, 0, VLD(from, 0));
                    VST(to, 1, VLD(from, 1));
                }
            }
        }
    } else {
        cfft_df_leaf_ps(n, x, tmp, k1, iw, wa, ifac, -1);
    }
}

/* Backward: the inverse of cfftf1_df_natural_ps. x holds the bins
   in natural order and is left with the sub-transform's result in natural
   order. deep > 0: the bins go to their regions in tmp, and the last pass
   moves the result back to x. */
static NEVER_INLINE(void) cfftb1_df_natural_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa,
                                               const int* ifac, int deep, const int* rev) {
    if (deep > 0) {
        const int idot = 2 * (n / 4), regions = 1 << (2 * deep), nleaf = n / regions;
        const int tile = regions < (PFFFT_DEPTH_FIRST_TILE) ? regions : (PFFFT_DEPTH_FIRST_TILE);
        v4sf* leaf_tmp = tmp + 2 * n;
        int j, q, c, c0;
        for (c0 = 0; c0 < regions; c0 += tile) {
            for (q = 0; q < nleaf; ++q) {
                const v4sf* from = x + 2 * (c0 + regions * q);
                for (c = c0; c < c0 + tile; ++c, from += 2) {
                    v4sf* to = tmp + 2 * (rev[c] * nleaf + q);
                    VST(to, 0, VLD(from, 0));
                    VST(to, 1, VLD(from, 1));
                }
            }
        }
        for (j = 0; j < 4; ++j) {
            cfft_df_backward_ps(n / 4, tmp + j * idot, leaf_tmp, k1 + 1, iw + 3 * idot, wa, ifac, deep - 1);
        }
        passb4_dit_op_ps(idot, tmp, x, &wa[iw], &wa[iw + idot], &wa[iw + 2 * idot]);
    } else {
        cfft_df_leaf_ps(n, x, tmp, k1, iw, wa, ifac, +1);
    }
}

/* Forward: the passes from factor index k1 on of the sub-transform of n
   elements in x: the next 'top' of them radix-4 passes in place, which
   leave 4^top slices, each then brought to natural order. The order of
   cfft_df_forward_ps. */
static NEVER_INLINE(void) cfftf1_df_slices_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa,
                                              const int* ifac, int top, int deep, const int* rev) {
    const int slices = 1 << (2 * top), nslice = n / slices;
    int slice, level;
    for (slice = 0; slice < slices; ++slice, x += 2 * nslice) {
        int idot = 2 * (n / 4), w = iw;
        for (level = 0; level < top; ++level, w += 3 * idot, idot /= 4) {
            if (slice % (slices >> (2 * level)) == 0) {
                passf4_ip_ps(idot, x, &wa[w], &wa[w + idot], &wa[w + 2 * idot], -1.f);
            }
        }
        cfftf1_df_natural_ps(nslice, x, tmp, k1 + top, w, wa, ifac, deep, rev);
    }
}

/* Backward: as cfftf1_df_slices_ps, the slices before the passes; the
   order of cfft_df_backward_ps. */
static NEVER_INLINE(void) cfftb1_df_slices_ps(int n, v4sf* x, v4sf* tmp, int k1, int iw, const float* wa,
                                              const int* ifac, int top, int deep, const int* rev) {
    const int slices = 1 << (2 * top), nslice = n / slices;
    /* the twiddles after those of the 'top' levels, as in
       cfft_df_backward_ps */
    const int w_slice = iw + 2 * (n - nslice);
    int slice, level;
    for (slice = 0; slice < slices; ++slice) {
        int idot = 2 * nslice, w = w_slice;
        cfftb1_df_natural_ps(nslice, x + 2 * slice * nslice, tmp, k1 + top, w_slice, wa, ifac, deep, rev);
        for (level = top - 1; level >= 0; --level, idot *= 4) {
            const int group = slices >> (2 * level); /* its slices */
            w -= 3 * idot;
            if ((slice + 1) % group == 0) {
                passb4_dit_ip_ps(idot, x + 2 * (slice + 1 - group) * nslice, &wa[w], &wa[w + idot], &wa[w + 2 * idot]);
            }
        }
    }
}

/* The passes of a forward transform of n interleaved elements in 'input',
   depth first: the result is in x, not input, as 4^top slices in natural
   order (see above). tmp is scratch for a slice and a leaf, 2 * n / 4^top
   and 2 * n / 4^(top + deep) vectors, and not x; it may be part of input,
   as the first pass has read the input before a slice uses it. */
static void
cfftf1_df_ps(int n, const v4sf* input, v4sf* x, v4sf* tmp, const float* wa, const int* ifac, int top, int deep,
             const int* rev) {
    const int idot = 2 * (n / 4);
    int j;
    assert(top >= 1 && ifac[2] == 4 && x != tmp && x != input);
    passf4_il_ps(idot, input, x, &wa[0], &wa[idot], &wa[2 * idot], -1.f);
    for (j = 0; j < 4; ++j) {
        cfftf1_df_slices_ps(n / 4, x + j * idot, tmp, 3, 3 * idot, wa, ifac, top - 1, deep, rev);
    }
}

/* The passes of a backward transform of the n bins in x (as cfftf1_df_ps
   leaves them), depth first, interleaved into out, not x; x is
   overwritten. tmp as for cfftf1_df_ps; it may be part of out, which the
   last pass alone stores to. */
static void
cfftb1_df_ps(int n, v4sf* x, v4sf* out, v4sf* tmp, const float* wa, const int* ifac, int top, int deep,
             const int* rev) {
    const int idot = 2 * (n / 4);
    int j;
    assert(top >= 1 && ifac[2] == 4 && x != tmp && x != out);
    for (j = 0; j < 4; ++j) {
        cfftb1_df_slices_ps(n / 4, x + j * idot, tmp, 3, 3 * idot, wa, ifac, top - 1, deep, rev);
    }
    passb4_dit_ol_ps(idot, x, out, &wa[0], &wa[idot], &wa[2 * idot]);
}
#endif /* PFFFT_DEPTH_FIRST */

#undef PF_INTERLEAVE2_ST
#undef PF_UNINTERLEAVE2_ST
#endif /* !PF_INPLACE_CODELETS */
#endif /* PF_FUSED_LAYOUT */
#if PFFFT_TERMINAL_RADIX8 && !PF_INPLACE_CODELETS
#undef PF_PASS8
#undef PF_PASS6
#endif
/* the codelet header macros, at the end of the inclusion with
   PF_INPLACE_CODELETS 0; the inclusion with 1 leaves them in their body
   form, which nothing after it uses. The including file undefines
   PF_INPLACE_CODELETS itself. */
#if !PF_INPLACE_CODELETS
#undef PF_CODELET_NAME
#undef PF_CODELET_RESTRICT
#undef PF_PASS_DIRECTIONS
#endif
