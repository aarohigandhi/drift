package io.drift.fmt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The published constants of each format, and the properties that must hold across
 * every code. The exhaustive cross-check against exact rational arithmetic lives in
 * python/verify_oracle.py; these are the checks that need nothing but Java.
 */
class MiniFloatTest {

    @Test
    void publishedRanges() {
        assertEquals(448.0, Formats.E4M3.maxFinite);
        assertEquals(0x1p-9, Formats.E4M3.minSubnormal);
        assertEquals(57344.0, Formats.E5M2.maxFinite);
        assertEquals(0x1p-16, Formats.E5M2.minSubnormal);
        assertEquals(6.0, Formats.E2M1.maxFinite);
        assertEquals(0.5, Formats.E2M1.minSubnormal);
        assertEquals(65504.0, Formats.FP16.maxFinite);
        assertEquals(Math.scalb(255.0, 120), Formats.BF16.maxFinite);
    }

    @Test
    void fp4HasTheEightStandardMagnitudes() {
        double[] expected = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        for (int c = 0; c < 8; c++) {
            assertEquals(expected[c], Formats.E2M1.decode(c));
            assertEquals(-expected[c], Formats.E2M1.decode(c | 8));
        }
    }

    @Test
    void everyCodeRoundTrips() {
        for (MiniFloat f : Formats.ALL) {
            for (int c = 0; c < f.codeCount(); c++) {
                double v = f.decode(c);
                // NaN has no single code to return to, and every format here saturates,
                // so an infinity deliberately re-encodes as the largest finite value.
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    continue;
                }
                assertEquals(c, f.encode(v), f.name + " code " + c);
            }
        }
    }

    @Test
    void specialValues() {
        // E4M3 has no infinity: its top-exponent codes other than all-ones are numbers.
        assertTrue(Double.isNaN(Formats.E4M3.decode(0x7F)));
        assertEquals(448.0, Formats.E4M3.decode(0x7E));
        assertEquals(Double.POSITIVE_INFINITY, Formats.E5M2.decode(0x7C));
        assertTrue(Double.isNaN(Formats.E5M2.decode(0x7D)));
    }

    @Test
    void tiesGoToEven() {
        assertEquals(2.0, Formats.E2M1.quantize(2.5));
        assertEquals(4.0, Formats.E2M1.quantize(3.5));
        assertEquals(1.0, Formats.E2M1.quantize(1.25));
        assertEquals(2.0, Formats.E2M1.quantize(1.75));
    }

    @Test
    void overflowSaturates() {
        assertEquals(448.0, Formats.E4M3.quantize(1e6));
        assertEquals(-448.0, Formats.E4M3.quantize(-1e6));
        assertEquals(6.0, Formats.E2M1.quantize(7.9));
        assertEquals(57344.0, Formats.E5M2.quantize(Double.POSITIVE_INFINITY));
    }

    @Test
    void negativeZeroSurvives() {
        int c = Formats.E4M3.encode(-0.0);
        assertEquals(0x80, c);
        assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(Formats.E4M3.decode(c)));
    }

    @Test
    void stochasticRoundingIsUnbiased() {
        Rng rng = new Rng(3);
        double sum = 0;
        int n = 400_000;
        for (int i = 0; i < n; i++) {
            sum += Formats.E2M1.quantize(2.25, Rounding.STOCHASTIC, rng);
        }
        // 2.25 sits a quarter of the way from 2 to 3; the standard error is ~7e-4.
        assertEquals(2.25, sum / n, 4e-3);
        assertEquals(2.0, Formats.E2M1.quantize(2.25, Rounding.NEAREST_EVEN, null));
    }
}
