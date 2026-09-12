package io.drift.kernel;

import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy.Scaling;

/**
 * Casting a tensor into a narrow format, and the scaling that has to happen first.
 *
 * <p>Everything here returns <em>effective</em> values: the number the hardware
 * would actually compute with, scale already applied. For MX that loses nothing,
 * because a shared exponent is a power of two and multiplying a representable value
 * by a power of two is exact, so every downstream product is bit-identical to a
 * kernel that kept the scale factored out. For per-tensor scaling the factor is an
 * arbitrary float, the multiply back is itself a rounding, and that rounding is part
 * of what the policy is measured on.
 */
public final class Quant {

    private Quant() {
    }

    /** Largest magnitude in a span. */
    public static double amax(double[] x, int off, int n) {
        double m = 0.0;
        for (int i = 0; i < n; i++) {
            double v = Math.abs(x[off + i]);
            if (v > m) {
                m = v;
            }
        }
        return m;
    }

    /** The factor that puts {@code amax} on the largest representable value. */
    public static double perTensorScale(MiniFloat f, double amax) {
        if (amax == 0.0 || !Double.isFinite(amax)) {
            return 1.0;
        }
        return f.maxFinite / amax;
    }

    /**
     * Cast a span into {@code f} under the policy's scaling, writing effective values.
     *
     * @param scale the per-tensor factor, used only by PER_TENSOR
     * @return the number of elements that hit the format maximum and were clamped
     */
    public static int quantize(double[] src, int srcOff, double[] dst, int dstOff, int n,
                               MiniFloat f, NumericPolicy p, double scale, Rng rng) {
        if (f == null) {
            System.arraycopy(src, srcOff, dst, dstOff, n);
            return 0;
        }
        Rounding rm = p.inputRounding;
        int clamped = 0;
        switch (p.scaling) {
            case NONE -> {
                for (int i = 0; i < n; i++) {
                    double v = src[srcOff + i];
                    if (Math.abs(v) > f.maxFinite) clamped++;
                    dst[dstOff + i] = f.quantize(v, rm, rng);
                }
            }
            case PER_TENSOR -> {
                double inv = 1.0 / scale;
                for (int i = 0; i < n; i++) {
                    double up = src[srcOff + i] * scale;
                    if (Math.abs(up) > f.maxFinite) clamped++;
                    dst[dstOff + i] = f.quantize(up, rm, rng) * inv;
                }
            }
            case MX_BLOCK -> clamped = quantizeMx(src, srcOff, dst, dstOff, n, f, p, rng);
        }
        return clamped;
    }

    /**
     * OCP Microscaling. Each block gets the shared exponent
     * {@code floor(log2(amax)) - emax_elem}, where emax_elem is the exponent of the
     * element format's largest normal, plus the policy's headroom.
     *
     * <p>With zero headroom that maps a block maximum into {@code [2^emax, 2^(emax+1))},
     * and every one of these formats stops short of the top of that range: e4m3 at 448
     * of [256, 512), e2m1 at 6 of [4, 8). So the largest element of a block clamps
     * whenever its mantissa is high enough, and it is always the element that
     * contributes most to any product.
     */
    private static int quantizeMx(double[] src, int srcOff, double[] dst, int dstOff, int n,
                                  MiniFloat f, NumericPolicy p, Rng rng) {
        int block = p.scaleBlock;
        int emaxElem = Math.getExponent(f.maxFinite);
        int clamped = 0;

        for (int start = 0; start < n; start += block) {
            int len = Math.min(block, n - start);
            double blockAmax = amax(src, srcOff + start, len);

            if (blockAmax == 0.0 || !Double.isFinite(blockAmax)) {
                for (int i = 0; i < len; i++) {
                    dst[dstOff + start + i] = 0.0;
                }
                continue;
            }

            int sharedExp = Math.getExponent(blockAmax) - emaxElem + p.mxHeadroom;
            sharedExp = Math.max(-127, Math.min(127, sharedExp));   // E8M0 range

            for (int i = 0; i < len; i++) {
                double scaled = Math.scalb(src[srcOff + start + i], -sharedExp);
                if (Math.abs(scaled) > f.maxFinite) clamped++;
                double q = f.quantize(scaled, p.inputRounding, rng);
                dst[dstOff + start + i] = Math.scalb(q, sharedExp);
            }
        }
        return clamped;
    }
}
