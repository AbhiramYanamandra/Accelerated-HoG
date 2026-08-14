// This file contains the fixed-point math helpers used to approximate square roots and trigonometric values efficiently for hardware-friendly HoG processing.
#ifndef FIXED_MATH_H
#define FIXED_MATH_H

#include <ap_fixed.h>
#include <ap_int.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <int W, int I>
ap_fixed<W, I> fixed_sqrt(ap_fixed<W, I> x)
{
    const int F = W - I;
    const int N = W + F;
    const int ITERS = (N + 1) / 2;
    const int P = 2 * ITERS;

    ap_uint<W> xraw = x.range(W - 1, 0);
    ap_uint<P> n = (ap_uint<P>)xraw << F;

    ap_uint<P + 2> rem = 0;
    ap_uint<ITERS> res = 0;

    for (int k = 0; k < ITERS; k++)
    {
        ap_uint<2> two = n.range(P - 1, P - 2);
        n <<= 2;
        rem = (rem << 2) | two;

        ap_uint<P + 2> trial = ((ap_uint<P + 2>)res << 2) | 1;
        if (rem >= trial)
        {
            rem -= trial;
            res = (res << 1) | 1;
        }
        else
        {
            res = res << 1;
        }
    }

    ap_ufixed<W, I> y;
    y.range(W - 1, 0) = (ap_uint<W>)res;
    return y;
}

template ap_fixed<64, 32> fixed_sqrt<64, 32>(ap_fixed<64, 32> x);
template ap_fixed<72, 40> fixed_sqrt<72, 40>(ap_fixed<72, 40> x);
template ap_fixed<32, 24> fixed_sqrt<32, 24>(ap_fixed<32, 24> x);
template ap_fixed<40, 24> fixed_sqrt<40, 24>(ap_fixed<40, 24> x);
template ap_fixed<48, 24> fixed_sqrt<48, 24>(ap_fixed<48, 24> x);

template <int W, int I>
ap_fixed<W, I> fixed_sqrt_signed(ap_fixed<W, I> x)
{
    if (x <= 0)
        return ap_ufixed<W, I>(0);
    ap_ufixed<W, I> ux;
    ux.range(W - 1, 0) = x.range(W - 1, 0);
    return fixed_sqrt<W, I>(ux);
}

typedef ap_fixed<24, 3> angle_t;

static const int CORDIC_ITERS = 16;

static const angle_t CORDIC_ATAN_LUT[CORDIC_ITERS] = {
    0.78539816339, 0.46364760900, 0.24497866313, 0.12435499455,
    0.06241880999, 0.03123983343, 0.01562372862, 0.00781234106,
    0.00390623013, 0.00195312252, 0.00097656219, 0.00048828121,
    0.00024414062, 0.00012207031, 0.00006103516, 0.00003051758};

template <int W, int I>
angle_t cordic_atan2(ap_fixed<W, I> y, ap_fixed<W, I> x)
{
    typedef ap_fixed<W + 4, I + 4> acc_t;

    acc_t xi = x;
    acc_t yi = y;
    angle_t correction = 0;

    if (x < 0)
    {
        xi = -xi;
        yi = -yi;
        correction = (y >= 0) ? angle_t(M_PI) : angle_t(-M_PI);
    }

    angle_t z = 0;
    for (int i = 0; i < CORDIC_ITERS; i++)
    {
#pragma HLS unroll
        acc_t xshift = xi >> i;
        acc_t yshift = yi >> i;
        if (yi >= 0)
        {
            xi = xi + yshift;
            yi = yi - xshift;
            z = z + CORDIC_ATAN_LUT[i];
        }
        else
        {
            xi = xi - yshift;
            yi = yi + xshift;
            z = z - CORDIC_ATAN_LUT[i];
        }
    }
    return z + correction;
}

template <int W, int I>
angle_t cordic_atan(ap_fixed<W, I> t)
{
    ap_fixed<W + 2, I + 2> tw = t;
    ap_fixed<W + 2, I + 2> one = 1;
    return cordic_atan2<W + 2, I + 2>(tw, one);
}

#endif