package io.drift.fmt;

import io.drift.fmt.MiniFloat.Overflow;
import io.drift.fmt.MiniFloat.Specials;

/**
 * The formats that actually appear in low precision training, as defined by IEEE 754
 * and by the OCP Microscaling specification.
 *
 * <p>FP32 is deliberately absent. It is the machine format, so simulating it would
 * add rounding that the hardware does not do; where a kernel wants FP32 it uses a
 * Java {@code float} and gets the real thing.
 */
public final class Formats {

    private Formats() {
    }

    /** OCP E4M3. Finite with NaN, no infinity. Max 448. The forward-pass FP8. */
    public static final MiniFloat E4M3 =
            new MiniFloat("e4m3", 4, 3, 7, Specials.FN, Overflow.SATURATE);

    /** OCP E5M2. IEEE-shaped, so it keeps Inf and NaN. Max 57344. The gradient FP8. */
    public static final MiniFloat E5M2 =
            new MiniFloat("e5m2", 5, 2, 15, Specials.IEEE, Overflow.SATURATE);

    /** FP4 as used by MXFP4 and NVFP4. Eight magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6. */
    public static final MiniFloat E2M1 =
            new MiniFloat("e2m1", 2, 1, 1, Specials.FINITE, Overflow.SATURATE);

    /** FP6, range-favouring variant. */
    public static final MiniFloat E3M2 =
            new MiniFloat("e3m2", 3, 2, 3, Specials.FINITE, Overflow.SATURATE);

    /** FP6, precision-favouring variant. */
    public static final MiniFloat E2M3 =
            new MiniFloat("e2m3", 2, 3, 1, Specials.FINITE, Overflow.SATURATE);

    /** IEEE binary16. */
    public static final MiniFloat FP16 =
            new MiniFloat("fp16", 5, 10, 15, Specials.IEEE, Overflow.SATURATE);

    /** bfloat16: FP32 range with seven mantissa bits. */
    public static final MiniFloat BF16 =
            new MiniFloat("bf16", 8, 7, 127, Specials.IEEE, Overflow.SATURATE);

    /** Every format above, for exhaustive sweeps. */
    public static final MiniFloat[] ALL = {E4M3, E5M2, E2M1, E3M2, E2M3, FP16, BF16};

    /** Look a format up by the name it reports. */
    public static MiniFloat byName(String name) {
        for (MiniFloat f : ALL) {
            if (f.name.equalsIgnoreCase(name)) {
                return f;
            }
        }
        throw new IllegalArgumentException("unknown format: " + name);
    }
}
