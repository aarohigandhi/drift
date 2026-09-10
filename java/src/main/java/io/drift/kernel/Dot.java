package io.drift.kernel;

import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy.Acc;

/**
 * The inner product, with the accumulator treated as an experimental variable.
 *
 * <p>The products are exact. That is not a simplification: a tensor core multiplies
 * two narrow inputs into a wide intermediate and rounds nothing, and for every
 * format here the exact product of two representable values fits in a double with
 * room to spare. So the only rounding in a dot product is the accumulation, which
 * is exactly the thing that is hard to vary on real hardware and easy to vary here.
 *
 * <p>One known limit. The running sum is held in a double, so an add whose two
 * operands are more than about 37 binades apart can round once into the double and
 * again into the accumulator format. Both roundings are to nearest, and the second
 * operand at that separation is already far below the accumulator quantum, so the
 * result differs from a single rounding only on exact ties. Nothing in the training
 * runs comes close to that spread; {@code FP64} and {@code FP32} are unaffected
 * either way, since for them the double is the arithmetic rather than an
 * intermediate.
 */
public final class Dot {

    private Dot() {
    }

    /**
     * Sum of {@code a[i] * b[i]} under the policy, over strided spans so a matmul can
     * walk a row against a column without copying either.
     */
    public static double dot(double[] a, int aOff, int aStride,
                             double[] b, int bOff, int bStride,
                             int n, NumericPolicy p, Rng rng) {
        return switch (p.order()) {
            case SEQUENTIAL -> straight(a, aOff, aStride, b, bOff, bStride, 0, n, p, rng);
            case REVERSED -> reversed(a, aOff, aStride, b, bOff, bStride, n, p, rng);
            case PAIRWISE -> pairwise(a, aOff, aStride, b, bOff, bStride, 0, n, p, rng);
            case BLOCKED -> blocked(a, aOff, aStride, b, bOff, bStride, n, p, rng);
        };
    }

    /** First to last, one running total. */
    private static double straight(double[] a, int aOff, int aStride,
                                   double[] b, int bOff, int bStride,
                                   int from, int count, NumericPolicy p, Rng rng) {
        Acc acc = p.acc();
        MiniFloat f = acc.format();
        Rounding rm = p.accRounding();

        if (acc == Acc.FP64) {
            double s = 0.0;
            for (int i = from; i < from + count; i++) {
                s += a[aOff + i * aStride] * b[bOff + i * bStride];
            }
            return s;
        }
        if (acc == Acc.FP32) {
            float s = 0.0f;
            for (int i = from; i < from + count; i++) {
                // One rounding per add, from the exact sum, which is what a tensor
                // core does. Rounding the product first would be a second one.
                s = (float) (s + a[aOff + i * aStride] * b[bOff + i * bStride]);
            }
            return s;
        }
        double s = 0.0;
        for (int i = from; i < from + count; i++) {
            s = f.quantize(s + a[aOff + i * aStride] * b[bOff + i * bStride], rm, rng);
        }
        return s;
    }

    /**
     * Last to first. The same products, the same format, the same accumulator width;
     * only the order differs, so whatever gap opens up between this and
     * {@link #straight} is error that no format can be blamed for.
     */
    private static double reversed(double[] a, int aOff, int aStride,
                                   double[] b, int bOff, int bStride,
                                   int n, NumericPolicy p, Rng rng) {
        Acc acc = p.acc();
        MiniFloat f = acc.format();
        Rounding rm = p.accRounding();

        if (acc == Acc.FP64) {
            double s = 0.0;
            for (int i = n - 1; i >= 0; i--) {
                s += a[aOff + i * aStride] * b[bOff + i * bStride];
            }
            return s;
        }
        if (acc == Acc.FP32) {
            float s = 0.0f;
            for (int i = n - 1; i >= 0; i--) {
                s = (float) (s + a[aOff + i * aStride] * b[bOff + i * bStride]);
            }
            return s;
        }
        double s = 0.0;
        for (int i = n - 1; i >= 0; i--) {
            s = f.quantize(s + a[aOff + i * aStride] * b[bOff + i * bStride], rm, rng);
        }
        return s;
    }

    /** Below this many terms, recursing costs more than it saves. */
    private static final int PAIRWISE_BASE = 8;

    /**
     * Recursive halving. Every partial sum is rounded, but the depth of the tree is
     * log n rather than n, so the errors compound over far fewer additions.
     */
    private static double pairwise(double[] a, int aOff, int aStride,
                                   double[] b, int bOff, int bStride,
                                   int from, int count, NumericPolicy p, Rng rng) {
        if (count <= PAIRWISE_BASE) {
            return straight(a, aOff, aStride, b, bOff, bStride, from, count, p, rng);
        }
        int half = count >>> 1;
        double left = pairwise(a, aOff, aStride, b, bOff, bStride, from, half, p, rng);
        double right = pairwise(a, aOff, aStride, b, bOff, bStride, from + half, count - half, p, rng);
        return addIn(left, right, p, rng);
    }

    /**
     * Chunks of {@code accBlock}, each summed in the accumulator width, with the
     * chunk totals summed in FP32.
     *
     * <p>This is what a real kernel does when it splits the K dimension across tiles
     * or across cooperating threads, and it is why the same matmul on the same
     * hardware can return a different number when only the tile size changes.
     */
    private static double blocked(double[] a, int aOff, int aStride,
                                  double[] b, int bOff, int bStride,
                                  int n, NumericPolicy p, Rng rng) {
        int block = Math.max(1, p.accBlock());
        float total = 0.0f;
        for (int start = 0; start < n; start += block) {
            int len = Math.min(block, n - start);
            double partial = straight(a, aOff, aStride, b, bOff, bStride, start, len, p, rng);
            total = (float) (total + partial);
        }
        return total;
    }

    /** A single add in the accumulator width, for combining partial sums. */
    private static double addIn(double x, double y, NumericPolicy p, Rng rng) {
        return switch (p.acc()) {
            case FP64 -> x + y;
            case FP32 -> (float) (x + y);
            case FP16, BF16 -> p.acc().format().quantize(x + y, p.accRounding(), rng);
        };
    }
}
