package io.drift.train;

import io.drift.fmt.Formats;
import io.drift.fmt.Rounding;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Acc;
import io.drift.kernel.NumericPolicy.Order;
import io.drift.kernel.NumericPolicy.Scaling;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The training sweep. Each policy differs from a named parent in one field, so any
 * gap between the two is attributable to that field.
 */
public final class Policies {

    private Policies() {
    }

    public static Map<String, NumericPolicy> all() {
        Map<String, NumericPolicy> m = new LinkedHashMap<>();

        m.put("exact", NumericPolicy.exact());

        NumericPolicy fp32 = NumericPolicy.builder("fp32").build();
        m.put("fp32", fp32);

        m.put("bf16", NumericPolicy.builder("bf16").input(Formats.BF16).build());

        // The usual FP8 recipe: E4M3 forward, E5M2 gradients, per-tensor current scaling.
        NumericPolicy fp8 = NumericPolicy.builder("fp8")
                .input(Formats.E4M3).grad(Formats.E5M2).scaling(Scaling.PER_TENSOR).build();
        m.put("fp8", fp8);

        // Format and scaling, one field at a time from fp8.
        m.put("fp8-delayed", fp8.toBuilder("fp8-delayed").amaxHistory(16).build());
        m.put("fp8-e4m3-grads", fp8.toBuilder("fp8-e4m3-grads").grad(Formats.E4M3).build());
        m.put("fp8-mx", fp8.toBuilder("fp8-mx").grad(Formats.E4M3).scaling(Scaling.MX_BLOCK).build());
        m.put("fp8-mx-headroom", fp8.toBuilder("fp8-mx-headroom").grad(Formats.E4M3)
                .scaling(Scaling.MX_BLOCK).mxHeadroom(1).build());

        // Summation, one field at a time from fp8. Cast held fixed.
        m.put("fp8-reversed", fp8.toBuilder("fp8-reversed").order(Order.REVERSED).build());
        m.put("fp8-acc-fp16", fp8.toBuilder("fp8-acc-fp16").acc(Acc.FP16).build());
        m.put("fp8-acc-fp16-pairwise", fp8.toBuilder("fp8-acc-fp16-pairwise")
                .acc(Acc.FP16).order(Order.PAIRWISE).build());
        m.put("fp8-acc-bf16", fp8.toBuilder("fp8-acc-bf16").acc(Acc.BF16).build());
        m.put("fp8-acc-bf16-pairwise", fp8.toBuilder("fp8-acc-bf16-pairwise")
                .acc(Acc.BF16).order(Order.PAIRWISE).build());
        m.put("fp8-acc-bf16-blocked", fp8.toBuilder("fp8-acc-bf16-blocked")
                .acc(Acc.BF16).order(Order.BLOCKED).accBlock(32).build());
        m.put("fp8-acc-bf16-sr", fp8.toBuilder("fp8-acc-bf16-sr")
                .acc(Acc.BF16).accRounding(Rounding.STOCHASTIC).build());

        // Four bit.
        NumericPolicy mxfp4 = NumericPolicy.builder("mxfp4")
                .input(Formats.E2M1).scaling(Scaling.MX_BLOCK).build();
        m.put("mxfp4", mxfp4);
        m.put("mxfp4-headroom", mxfp4.toBuilder("mxfp4-headroom").mxHeadroom(1).build());
        m.put("mxfp4-sr", mxfp4.toBuilder("mxfp4-sr").inputRounding(Rounding.STOCHASTIC).build());

        return m;
    }

    public static NumericPolicy byName(String name) {
        NumericPolicy p = all().get(name);
        if (p == null) {
            throw new IllegalArgumentException("unknown policy " + name + "; known: " + all().keySet());
        }
        return p;
    }
}
