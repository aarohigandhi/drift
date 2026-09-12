package io.drift.kernel;

import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy.Acc;

/**
 * The inner product, with the accumulator treated as an experimental variable.
 *
 * <p>The products are exact. A tensor core multiplies two narrow inputs into a wide
 * intermediate and rounds nothing, and for every format here the exact product of
 * two representable values fits in a double. So the only rounding in a dot product
 * is the accumulation, which is the part that cannot be varied on real hardware.
 *
 * <p>One known limit. The running sum lives in a double, so an add whose operands are
 * more than about 37 binades apart can round once into the double and once more into
 * a narrow accumulator. Both roundings are to nearest and the smaller operand is then
 * far below the accumulator quantum, so the result can differ from a single rounding
 * only on an exact tie. FP64 and FP32 are unaffected.
 */
public final class Dot {

    private Dot() {
    }

    public static double dot(double[] a, int aOff, int aStride,
                             double[] b, int bOff, int bStride,
                             int n, NumericPolicy p, Rng rng) {
        return switch (p.order) {
            case SEQUENTIAL -> forward(a, aOff, aStride, b, bOff, bStride, 0, n, p, rng);
            case REVERSED -> backward(a, aOff, aStride, b, bOff, bStride, n, p, rng);
            case PAIRWISE -> pairwise(a, aOff, aStride, b, bOff, bStride, 0, n, p, rng);
            case BLOCKED -> blocked(a, aOff, aStride, b, bOff, bStride, n, p, rng);
        };
    }

    private static double forward(double[] a, int aOff, int aStride,
                                  double[] b, int bOff, int bStride,
                                  int from, int count, NumericPolicy p, Rng rng) {
        int ia = aOff + from * aStride;
        int ib = bOff + from * bStride;
        switch (p.acc) {
            case FP64: {
                double s = 0.0;
                for (int i = 0; i < count; i++, ia += aStride, ib += bStride) {
                    s += a[ia] * b[ib];
                }
                return s;
            }
            case FP32: {
                float s = 0.0f;
                for (int i = 0; i < count; i++, ia += aStride, ib += bStride) {
                    // One rounding per add, from the exact sum, as a tensor core does.
                    s = (float) (s + a[ia] * b[ib]);
                }
                return s;
            }
            default: {
                Acc acc = p.acc;
                Rounding rm = p.accRounding;
                double s = 0.0;
                for (int i = 0; i < count; i++, ia += aStride, ib += bStride) {
                    s = round(s + a[ia] * b[ib], acc, rm, rng);
                }
                return s;
            }
        }
    }

    private static double backward(double[] a, int aOff, int aStride,
                                   double[] b, int bOff, int bStride,
                                   int n, NumericPolicy p, Rng rng) {
        double s = 0.0;
        float f = 0.0f;
        for (int i = n - 1; i >= 0; i--) {
            double prod = a[aOff + i * aStride] * b[bOff + i * bStride];
            switch (p.acc) {
                case FP64 -> s += prod;
                case FP32 -> f = (float) (f + prod);
                default -> s = round(s + prod, p.acc, p.accRounding, rng);
            }
        }
        return p.acc == Acc.FP32 ? f : s;
    }

    private static final int PAIRWISE_BASE = 8;

    private static double pairwise(double[] a, int aOff, int aStride,
                                   double[] b, int bOff, int bStride,
                                   int from, int count, NumericPolicy p, Rng rng) {
        if (count <= PAIRWISE_BASE) {
            return forward(a, aOff, aStride, b, bOff, bStride, from, count, p, rng);
        }
        int half = count >>> 1;
        double left = pairwise(a, aOff, aStride, b, bOff, bStride, from, half, p, rng);
        double right = pairwise(a, aOff, aStride, b, bOff, bStride, from + half, count - half, p, rng);
        return switch (p.acc) {
            case FP64 -> left + right;
            case FP32 -> (float) (left + right);
            default -> round(left + right, p.acc, p.accRounding, rng);
        };
    }

    private static double blocked(double[] a, int aOff, int aStride,
                                  double[] b, int bOff, int bStride,
                                  int n, NumericPolicy p, Rng rng) {
        int block = Math.max(1, p.accBlock);
        if (p.acc == Acc.FP64) {
            double total = 0.0;
            for (int start = 0; start < n; start += block) {
                total += forward(a, aOff, aStride, b, bOff, bStride, start, Math.min(block, n - start), p, rng);
            }
            return total;
        }
        float total = 0.0f;
        for (int start = 0; start < n; start += block) {
            double partial = forward(a, aOff, aStride, b, bOff, bStride, start, Math.min(block, n - start), p, rng);
            total = (float) (total + partial);
        }
        return total;
    }

    // ------------------------------------------------------------ fast rounding

    private static final int FP16_MANT = 10;
    private static final int FP16_MIN_EXP = -14;
    private static final double FP16_MAX = 65504.0;

    private static final int BF16_MANT = 7;
    private static final int BF16_MIN_EXP = -126;
    private static final double BF16_MAX = Math.scalb(255.0, 120);

    /**
     * Round a finite value into FP16 or BF16, with saturation.
     *
     * <p>The same arithmetic as {@link io.drift.fmt.MiniFloat#quantize}, without the
     * trip through a bit pattern, because it runs once per multiply-accumulate. It
     * consumes random numbers in exactly the same pattern, so a test can hold the two
     * to identical outputs under stochastic rounding as well as nearest.
     */
    public static double round(double x, Acc acc, Rounding rm, Rng rng) {
        int mant;
        int minExp;
        double max;
        if (acc == Acc.FP16) {
            mant = FP16_MANT;
            minExp = FP16_MIN_EXP;
            max = FP16_MAX;
        } else {
            mant = BF16_MANT;
            minExp = BF16_MIN_EXP;
            max = BF16_MAX;
        }
        if (x == 0.0 || Double.isNaN(x)) {
            return x;
        }
        double ax = Math.abs(x);
        if (Double.isInfinite(ax)) {
            return Math.copySign(max, x);
        }
        int e = Math.getExponent(ax);
        if (e < minExp) {
            e = minExp;
        }
        int q = e - mant;
        double scaled = Math.scalb(ax, -q);
        double m;
        switch (rm) {
            case NEAREST_EVEN -> m = Math.rint(scaled);
            case TRUNCATE -> m = Math.floor(scaled);
            default -> {
                double fl = Math.floor(scaled);
                double frac = scaled - fl;
                m = frac == 0.0 ? fl : fl + (rng.nextDouble() < frac ? 1 : 0);
            }
        }
        double r = Math.scalb(m, q);
        if (r > max) {
            r = max;
        }
        return Math.copySign(r, x);
    }
}
