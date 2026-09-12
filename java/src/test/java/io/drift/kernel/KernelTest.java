package io.drift.kernel;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy.Acc;
import io.drift.kernel.NumericPolicy.Order;
import io.drift.kernel.NumericPolicy.Scaling;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KernelTest {

    /**
     * The accumulator uses its own rounding for speed. It has to be the same function
     * as MiniFloat, which the oracle verified exhaustively, down to the bit and down to
     * the random numbers it consumes.
     */
    @Test
    void fastAccumulatorRoundingMatchesMiniFloat() {
        Random r = new Random(11);
        for (Acc acc : new Acc[]{Acc.FP16, Acc.BF16}) {
            MiniFloat f = acc.format();
            for (Rounding rm : Rounding.values()) {
                Rng fast = new Rng(99);
                Rng slow = new Rng(99);
                for (int i = 0; i < 1_000_000; i++) {
                    // Span far below the smallest subnormal to far above the maximum.
                    double x = (r.nextBoolean() ? 1 : -1)
                            * Math.scalb(1 + r.nextDouble(), r.nextInt(300) - 150);
                    if (i % 97 == 0) {
                        x = Math.rint(x);   // exercise exact ties
                    }
                    double a = Dot.round(x, acc, rm, fast);
                    double b = f.quantize(x, rm, slow);
                    assertEquals(Double.doubleToRawLongBits(b), Double.doubleToRawLongBits(a),
                            acc + " " + rm + " x=" + x);
                }
                assertEquals(slow.state(), fast.state(), "random streams diverged for " + acc + " " + rm);
            }
        }
    }

    @Test
    void gradientRoundingDefaultsToInputRoundingAndCanDiffer() {
        NumericPolicy same = NumericPolicy.builder("s").input(Formats.E2M1)
                .inputRounding(Rounding.STOCHASTIC).build();
        assertEquals(Rounding.STOCHASTIC, same.gradCastRounding());
        NumericPolicy split = same.toBuilder("t").gradRounding(Rounding.NEAREST_EVEN).build();
        assertEquals(Rounding.STOCHASTIC, split.inputRounding);
        assertEquals(Rounding.NEAREST_EVEN, split.gradCastRounding());
    }

    @Test
    void everyOrderAgreesInFp64() {
        Random r = new Random(5);
        int n = 1000;
        double[] a = new double[n];
        double[] b = new double[n];
        double plain = 0;
        for (int i = 0; i < n; i++) {
            a[i] = r.nextGaussian();
            b[i] = r.nextGaussian();
            plain += a[i] * b[i];
        }
        for (Order o : Order.values()) {
            NumericPolicy p = NumericPolicy.builder("t").acc(Acc.FP64).order(o).accBlock(37).build();
            assertEquals(plain, Dot.dot(a, 0, 1, b, 0, 1, n, p, null), 1e-12 * Math.abs(plain), o.name());
        }
    }

    /** Float addition is not associative, and the kernel must expose that rather than hide it. */
    @Test
    void summationOrderChangesTheAnswerInFp32() {
        double[] a = {1.0, 1e8, -1e8};
        double[] ones = {1.0, 1.0, 1.0};
        NumericPolicy seq = NumericPolicy.builder("s").acc(Acc.FP32).order(Order.SEQUENTIAL).build();
        NumericPolicy rev = NumericPolicy.builder("r").acc(Acc.FP32).order(Order.REVERSED).build();
        assertEquals(0.0, Dot.dot(a, 0, 1, ones, 0, 1, 3, seq, null));
        assertEquals(1.0, Dot.dot(a, 0, 1, ones, 0, 1, 3, rev, null));
    }

    @Test
    void stridesWalkColumns() {
        // 2x3 matrix, column 1 is {2, 5}.
        double[] m = {1, 2, 3, 4, 5, 6};
        double[] v = {10, 100};
        NumericPolicy p = NumericPolicy.exact();
        assertEquals(2 * 10 + 5 * 100, Dot.dot(m, 1, 3, v, 0, 1, 2, p, null));
    }

    @Test
    void mxScalesArePowersOfTwoSoDequantisationIsExact() {
        NumericPolicy p = NumericPolicy.builder("mx").input(Formats.E2M1).scaling(Scaling.MX_BLOCK).build();
        Random r = new Random(2);
        double[] x = new double[256];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextGaussian() * 3;
        }
        double[] q = new double[x.length];
        Quant.quantize(x, 0, q, 0, x.length, Formats.E2M1, p, 1.0, null);
        for (int start = 0; start < x.length; start += 32) {
            double amax = Quant.amax(x, start, 32);
            int shared = Math.getExponent(amax) - 2;
            for (int i = start; i < start + 32; i++) {
                double element = Math.scalb(q[i], -shared);
                assertEquals(element, Formats.E2M1.quantize(element), "element must be an FP4 value");
            }
        }
    }

    /**
     * The OCP shared exponent maps a block maximum into [4, 8) for FP4, which stops at
     * 6, so a block whose maximum has a high mantissa clamps. One bit of headroom
     * removes it.
     */
    @Test
    void specExponentClampsAndHeadroomDoesNot() {
        double[] block = new double[32];
        block[0] = 7.0;
        block[1] = 0.3;
        double[] q = new double[32];
        NumericPolicy spec = NumericPolicy.builder("s").input(Formats.E2M1).scaling(Scaling.MX_BLOCK).build();
        NumericPolicy head = spec.toBuilder("h").mxHeadroom(1).build();
        assertEquals(1, Quant.quantize(block, 0, q, 0, 32, Formats.E2M1, spec, 1.0, null));
        assertEquals(6.0, q[0]);
        assertEquals(0, Quant.quantize(block, 0, q, 0, 32, Formats.E2M1, head, 1.0, null));
        assertEquals(8.0, q[0]);
    }

    @Test
    void narrowAccumulatorsLoseMoreSequentiallyThanPairwise() {
        Random r = new Random(8);
        int n = 4096;
        int worse = 0;
        int trials = 50;
        for (int t = 0; t < trials; t++) {
            double[] a = new double[n];
            double[] b = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
                b[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
            }
            double exact = Dot.dot(a, 0, 1, b, 0, 1, n, NumericPolicy.exact(), null);
            NumericPolicy seq = NumericPolicy.builder("s").acc(Acc.BF16).build();
            NumericPolicy pair = NumericPolicy.builder("p").acc(Acc.BF16).order(Order.PAIRWISE).build();
            double es = Math.abs(Dot.dot(a, 0, 1, b, 0, 1, n, seq, null) - exact);
            double ep = Math.abs(Dot.dot(a, 0, 1, b, 0, 1, n, pair, null) - exact);
            if (es > ep) {
                worse++;
            }
        }
        assertTrue(worse > trials * 0.8, "sequential worse in " + worse + " of " + trials);
        assertNotEquals(0, worse);
    }
}
