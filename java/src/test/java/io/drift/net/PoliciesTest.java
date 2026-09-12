package io.drift.net;

import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy;
import io.drift.train.Policies;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoliciesTest {

    /** The README's main table is 18 policies; follow-up experiments must not leak into it. */
    @Test
    void mainSweepIsTheEighteenInTheReadme() {
        assertEquals(18, Policies.mainSweep().size());
        assertFalse(Policies.mainSweep().containsKey("acc-bf16"));
        assertTrue(Policies.all().containsKey("acc-bf16"));
    }

    @Test
    void stochasticRoundingSplitChangesOnlyOneHalf() {
        NumericPolicy fwd = Policies.byName("mxfp4-sr-fwd");
        NumericPolicy bwd = Policies.byName("mxfp4-sr-bwd");
        assertEquals(Rounding.STOCHASTIC, fwd.inputRounding);
        assertEquals(Rounding.NEAREST_EVEN, fwd.gradCastRounding());
        assertEquals(Rounding.NEAREST_EVEN, bwd.inputRounding);
        assertEquals(Rounding.STOCHASTIC, bwd.gradCastRounding());
    }

    /** The uncast accumulator runs differ from fp32 in the accumulator and nothing else. */
    @Test
    void uncastAccumulatorPoliciesOnlyChangeTheAccumulator() {
        NumericPolicy base = Policies.byName("fp32");
        for (String name : new String[]{"acc-fp16", "acc-fp16-pairwise", "acc-bf16", "acc-bf16-pairwise", "acc-bf16-blocked"}) {
            NumericPolicy p = Policies.byName(name);
            assertEquals(null, p.inputFormat, name);
            assertEquals(base.scaling, p.scaling, name);
            assertEquals(base.masterFp32, p.masterFp32, name);
            assertTrue(p.acc != base.acc, name);
        }
    }
}
