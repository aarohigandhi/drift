package io.drift.kernel;

import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.kernel.NumericPolicy.Scaling;

/**
 * Casting a tensor into a narrow format, and the scaling that has to happen first.
 *
 * <p>A tensor is quantised once and then used across a whole matmul, which is what
 * hardware does and also the only thing that makes the cost bearable.
 *
 * <p>Everything here returns <em>effective</em> values: the number the hardware
 * would actually be computing with, scale already applied. For {@link
 * Scaling#MX_BLOCK} that loses nothing, because a shared exponent is a power of two
 * and multiplying a representable value by a power of two is exact, so the products
 * downstream are bit-identical to what a kernel that kept the scale factored out
 * would produce. For {@link Scaling#PER_TENSOR} the factor is an arbitrary float,
 * the multiply back is itself a rounding, and that rounding is part of what the
 * policy is being measured on.
 */
public final class Quant {

    private Quant() {
    }

    /** Largest magnitude in a span. Zero if the span is all zeros. */
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

    /**
     * The per-tensor factor that puts the largest element on the largest
     * representable value. Returns 1 for an all-zero tensor.
     *
     * <p>Kept separate from the cast so that a caller can hand back a factor
     * computed several steps ago, which is what delayed scaling is and where its
     * failure mode comes from.
     */
    public static double perTensorScale(MiniFloat f, double amax) {
        if (amax == 0.0 || !Double.isFinite(amax)) {
            return 1.0;
        }
        return f.maxFinite / amax;
    }

    /**
     * Cast a span into the policy's format, writing effective values into {@code dst}.
     *
     * @param scale the per-tensor factor, used only by {@link Scaling#PER_TENSOR};
     *              pass the value from {@link #perTensorScale} for current scaling or
     *              a stale one for delayed
     */
    public static void quantize(double[] src, int srcOff, double[] dst, int dstOff, int n,
                                NumericPolicy p, double scale, Rng rng) {
        MiniFloat f = p.inputFormat();
        if (f == null) {
            System.arraycopy(src, srcOff, dst, dstOff, n);
            return;
        }
        switch (p.scaling()) {
            case NONE -> {
                for (int i = 0; i < n; i++) {
                    dst[dstOff + i] = f.quantize(src[srcOff + i], p.inputRounding(), rng);
                }
            }
            case PER_TENSOR -> {
                double inv = 1.0 / scale;
                for (int i = 0; i < n; i++) {
                    double up = src[srcOff + i] * scale;
                    dst[dstOff + i] = f.quantize(up, p.inputRounding(), rng) * inv;
                }
            }
            case MX_BLOCK -> quantizeMx(src, srcOff, dst, dstOff, n, f, p, rng);
        }
    }

    /**
     * OCP Microscaling. Each block of {@code scaleBlock} elements gets one shared
     * exponent, held in E8M0, chosen so the block maximum lands in the top binade of
     * the element format.
     */
    private static void quantizeMx(double[] src, int srcOff, double[] dst, int dstOff, int n,
                                   MiniFloat f, NumericPolicy p, Rng rng) {
        int block = p.scaleBlock();
        int emaxElem = Math.getExponent(f.maxFinite);

        for (int start = 0; start < n; start += block) {
            int len = Math.min(block, n - start);
            double blockAmax = amax(src, srcOff + start, len);

            if (blockAmax == 0.0 || !Double.isFinite(blockAmax)) {
                for (int i = 0; i < len; i++) {
                    dst[dstOff + start + i] = 0.0;
                }
                continue;
            }

            int sharedExp = Math.getExponent(blockAmax) - emaxElem;
            // E8M0 holds a biased exponent in eight bits, so the shared factor cannot
            // leave this range whatever the tensor does.
            sharedExp = Math.max(-127, Math.min(127, sharedExp));

            for (int i = 0; i < len; i++) {
                // Both rescalings are powers of two, so neither loses anything and
                // the only error in this loop is the cast in the middle.
                double scaled = Math.scalb(src[srcOff + start + i], -sharedExp);
                double q = f.quantize(scaled, p.inputRounding(), rng);
                dst[dstOff + start + i] = Math.scalb(q, sharedExp);
            }
        }
    }
}
