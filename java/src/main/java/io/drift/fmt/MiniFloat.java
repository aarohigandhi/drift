package io.drift.fmt;

/**
 * A binary floating point format of arbitrary width, defined by its exponent and
 * mantissa field sizes and by what it does with the top exponent.
 *
 * <p>Every format used in low precision training is an instance of this: FP16 and
 * BF16 are IEEE-shaped, OCP E4M3 gives up infinity to buy one more binade, and FP4
 * E2M1 has no special values at all because with sixteen codes it cannot spare one.
 * Writing them once, parameterised, means the rounding path is a single piece of
 * code that gets exhaustively tested rather than five near-copies that do not.
 *
 * <p>The rounding is exact rather than approximate. The quantum at a given exponent
 * is a power of two, so dividing by it only shifts the exponent field of a double
 * and loses nothing; the scaled value is then an integer plus a fraction that a
 * double represents exactly. That is what lets {@code io.drift.tools.DumpTables} and
 * {@code python/verify_oracle.py} agree to the bit over every code in the format.
 */
public final class MiniFloat {

    /** What the all-ones exponent means. */
    public enum Specials {
        /** IEEE 754: top exponent is Inf (mantissa 0) or NaN (mantissa non-zero). */
        IEEE,
        /**
         * OCP "FN": finite, with NaN. Only the all-ones exponent with all-ones
         * mantissa is NaN. There is no infinity, which buys back every other code in
         * the top binade. That is how E4M3 reaches 448 instead of 240.
         */
        FN,
        /** No infinity and no NaN. Every bit pattern is a finite number. FP4 E2M1. */
        FINITE
    }

    /** What happens to a value larger than the format can hold. */
    public enum Overflow {
        /**
         * Clamp to the largest finite value. What every FP8 training recipe does,
         * because the alternative poisons the whole tensor.
         */
        SATURATE,
        /** Produce Inf or NaN as the format allows. IEEE behaviour. */
        PROPAGATE
    }

    public final String name;
    public final int expBits;
    public final int mantBits;
    public final int bias;
    public final Specials specials;
    public final Overflow overflow;

    /** Total width in bits, including the sign. */
    public final int width;
    private final int mantMask;
    private final int expMask;
    private final int signBit;
    private final int maxExpField;
    private final int codeMask;

    /** Unbiased exponent of the smallest normal value. */
    public final int minNormalExp;
    /** Largest finite magnitude. */
    public final double maxFinite;
    /** Smallest positive normal magnitude. */
    public final double minNormal;
    /** Smallest positive subnormal magnitude, the quantum in the subnormal range. */
    public final double minSubnormal;

    public MiniFloat(String name, int expBits, int mantBits, int bias,
                     Specials specials, Overflow overflow) {
        if (expBits < 1 || mantBits < 0 || expBits + mantBits > 30) {
            throw new IllegalArgumentException("unsupported field widths");
        }
        this.name = name;
        this.expBits = expBits;
        this.mantBits = mantBits;
        this.bias = bias;
        this.specials = specials;
        this.overflow = overflow;

        this.width = 1 + expBits + mantBits;
        this.mantMask = (1 << mantBits) - 1;
        this.maxExpField = (1 << expBits) - 1;
        this.expMask = maxExpField << mantBits;
        this.signBit = 1 << (expBits + mantBits);
        this.codeMask = (1 << width) - 1;

        this.minNormalExp = 1 - bias;
        this.minNormal = Math.scalb(1.0, minNormalExp);
        this.minSubnormal = Math.scalb(1.0, minNormalExp - mantBits);

        // The largest finite code depends on how much of the top binade the format
        // gave away to special values.
        int topExpField;
        int topMant;
        switch (specials) {
            case IEEE -> { topExpField = maxExpField - 1; topMant = mantMask; }
            case FN -> { topExpField = maxExpField; topMant = mantMask - 1; }
            case FINITE -> { topExpField = maxExpField; topMant = mantMask; }
            default -> throw new IllegalStateException();
        }
        if (topMant < 0) {
            throw new IllegalArgumentException("format too narrow for its special values");
        }
        this.maxFinite = Math.scalb(1.0 + (double) topMant / (1 << mantBits), topExpField - bias);
    }

    /** Number of distinct bit patterns, both signed zeros included. */
    public int codeCount() {
        return 1 << width;
    }

    // ---------------------------------------------------------------- decode

    /** The real value of a bit pattern. Patterns wider than the format are rejected. */
    public double decode(int bits) {
        if ((bits & ~codeMask) != 0) {
            throw new IllegalArgumentException("bits outside format width: " + bits);
        }
        int expField = (bits & expMask) >>> mantBits;
        int mant = bits & mantMask;
        double sign = (bits & signBit) != 0 ? -1.0 : 1.0;

        if (expField == maxExpField) {
            switch (specials) {
                case IEEE -> {
                    return mant == 0 ? sign * Double.POSITIVE_INFINITY : Double.NaN;
                }
                case FN -> {
                    if (mant == mantMask) return Double.NaN;
                }
                case FINITE -> { /* an ordinary number */ }
            }
        }
        if (expField == 0) {
            // Subnormal: no implicit leading one, exponent pinned at the minimum.
            return sign * Math.scalb((double) mant, minNormalExp - mantBits);
        }
        double significand = 1.0 + (double) mant / (1 << mantBits);
        return sign * Math.scalb(significand, expField - bias);
    }

    // ---------------------------------------------------------------- encode

    /** Encode with round-to-nearest-even. */
    public int encode(double v) {
        return encode(v, Rounding.NEAREST_EVEN, null);
    }

    /**
     * Encode a real value into this format.
     *
     * @param rng required for {@link Rounding#STOCHASTIC}, unused otherwise
     */
    public int encode(double v, Rounding rm, Rng rng) {
        if (Double.isNaN(v)) {
            if (specials == Specials.FINITE) {
                throw new ArithmeticException(name + " cannot represent NaN");
            }
            return nanBits();
        }

        int sign = (v < 0.0 || (v == 0.0 && 1.0 / v < 0.0)) ? signBit : 0;
        double a = Math.abs(v);

        if (a == 0.0) {
            return sign;
        }
        if (Double.isInfinite(a)) {
            return sign | overflowBits();
        }

        // Pick the binade, then the quantum inside it. Below the smallest normal the
        // quantum stops shrinking, which is the whole point of subnormals.
        int e = Math.getExponent(a);
        if (e == Double.MIN_EXPONENT - 1 || e < minNormalExp) {
            // Either a subnormal double, where the exponent query does not mean what
            // it usually does, or a value below this format's smallest normal. Both
            // are encoded against the fixed subnormal quantum.
            e = minNormalExp;
        }

        int quantumExp = e - mantBits;
        double scaled = Math.scalb(a, -quantumExp);   // exact: a power-of-two rescale
        long m = roundToIntegral(scaled, rm, rng);

        long normalBase = 1L << mantBits;

        if (m < normalBase) {
            // Subnormal. The raw integer is the encoding: if rounding pushed it up to
            // exactly normalBase it carries into the exponent field by itself, which
            // is the usual IEEE encoding trick and is handled by the branch below.
            return sign | (int) m;
        }
        if (m >= (normalBase << 1)) {
            // Rounding carried out of the binade. It can only land exactly on the
            // boundary, so halving is exact.
            m >>= 1;
            e += 1;
        }

        int expField = e + bias;
        int mantField = (int) (m - normalBase);

        if (expField > maxExpField
                || (expField == maxExpField && !topBinadeAllows(mantField))) {
            return sign | overflowBits();
        }
        return sign | (expField << mantBits) | mantField;
    }

    /** Round trip: the value this format would actually store for {@code v}. */
    public double quantize(double v) {
        return decode(encode(v, Rounding.NEAREST_EVEN, null));
    }

    /** Round trip under an explicit rounding mode. */
    public double quantize(double v, Rounding rm, Rng rng) {
        return decode(encode(v, rm, rng));
    }

    private boolean topBinadeAllows(int mantField) {
        return switch (specials) {
            case IEEE -> false;                  // top exponent is Inf/NaN only
            case FN -> mantField < mantMask;     // all-ones mantissa is NaN
            case FINITE -> true;
        };
    }

    private int nanBits() {
        return switch (specials) {
            case IEEE -> expMask | 1;
            case FN -> expMask | mantMask;
            case FINITE -> throw new ArithmeticException(name + " has no NaN");
        };
    }

    private int overflowBits() {
        if (overflow == Overflow.SATURATE) {
            return maxFiniteBits();
        }
        return switch (specials) {
            case IEEE -> expMask;                // +Inf
            case FN -> expMask | mantMask;       // no Inf exists, so NaN
            case FINITE -> maxFiniteBits();      // nothing to propagate to
        };
    }

    private int maxFiniteBits() {
        return switch (specials) {
            case IEEE -> ((maxExpField - 1) << mantBits) | mantMask;
            case FN -> (maxExpField << mantBits) | (mantMask - 1);
            case FINITE -> (maxExpField << mantBits) | mantMask;
        };
    }

    private static long roundToIntegral(double scaled, Rounding rm, Rng rng) {
        switch (rm) {
            case NEAREST_EVEN:
                return (long) Math.rint(scaled);
            case TRUNCATE:
                return (long) Math.floor(scaled);
            case STOCHASTIC: {
                double fl = Math.floor(scaled);
                double frac = scaled - fl;
                if (frac == 0.0) {
                    return (long) fl;
                }
                if (rng == null) {
                    throw new IllegalArgumentException("stochastic rounding needs an Rng");
                }
                return (long) fl + (rng.nextDouble() < frac ? 1 : 0);
            }
            default:
                throw new IllegalStateException();
        }
    }

    @Override
    public String toString() {
        return name;
    }
}
