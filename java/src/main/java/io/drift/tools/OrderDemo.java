package io.drift.tools;

import io.drift.fmt.Formats;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.Dot;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Acc;
import io.drift.kernel.NumericPolicy.Order;
import io.drift.kernel.NumericPolicy.Scaling;
import io.drift.kernel.Quant;

import java.util.Locale;
import java.util.Random;

/**
 * One dot product, computed several ways, to check the kernel measures anything at
 * all before it is wired into a training loop.
 *
 * <p>The two halves of the table answer different questions. The first varies the
 * format and holds the summation fixed. The second holds the format fixed and
 * varies only the order the products are added in, which is the comparison that
 * cannot be run on real hardware.
 */
public final class OrderDemo {

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        long seed = args.length > 1 ? Long.parseLong(args[1]) : 7;

        // Activations are not uniform. A normal draw times a lognormal scale gives
        // the heavy tail and the few binades of spread that real tensors have, which
        // is what makes summation order matter in the first place.
        Random r = new Random(seed);
        double[] a = new double[n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
            b[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
        }

        double truth = Dot.dot(a, 0, 1, b, 0, 1, n, NumericPolicy.exact(), null);

        System.out.printf(Locale.ROOT, "n = %d, exact fp64 value = %.10e%n%n", n, truth);

        System.out.println("format varied, summation held at fp32 sequential");
        System.out.println("  policy                       relative error");
        row("bf16", NumericPolicy.bf16Baseline(), a, b, n, truth);
        row("fp8 e4m3, per-tensor", NumericPolicy.fp8Baseline(), a, b, n, truth);
        row("mxfp4", NumericPolicy.mxfp4(), a, b, n, truth);

        System.out.println();
        System.out.println("format held at bf16, only the summation varied");
        System.out.println("  policy                       relative error");
        NumericPolicy base = NumericPolicy.bf16Baseline();
        row("fp32 sequential", base, a, b, n, truth);
        row("fp32 reversed", base.withOrder("r", Order.REVERSED, 128), a, b, n, truth);
        row("fp32 pairwise", base.withOrder("p", Order.PAIRWISE, 128), a, b, n, truth);
        row("fp32 blocked, k=32", base.withOrder("b32", Order.BLOCKED, 32), a, b, n, truth);
        row("fp32 blocked, k=128", base.withOrder("b128", Order.BLOCKED, 128), a, b, n, truth);
        row("fp16 sequential", base.with("f16", Acc.FP16, Order.SEQUENTIAL), a, b, n, truth);
        row("fp16 pairwise", base.with("f16p", Acc.FP16, Order.PAIRWISE), a, b, n, truth);
        row("bf16 sequential", base.with("b16", Acc.BF16, Order.SEQUENTIAL), a, b, n, truth);
        row("bf16 pairwise", base.with("b16p", Acc.BF16, Order.PAIRWISE), a, b, n, truth);

        System.out.println();
        System.out.println("fp8 e4m3 inputs, scaling policy varied, fp32 sequential");
        System.out.println("  policy                       relative error");
        NumericPolicy f8 = NumericPolicy.fp8Baseline();
        row("no scaling", f8.withScaling("none", Scaling.NONE, 32), a, b, n, truth);
        row("per-tensor", f8, a, b, n, truth);
        row("mx block, 32", f8.withScaling("mx32", Scaling.MX_BLOCK, 32), a, b, n, truth);
        row("mx block, 128", f8.withScaling("mx128", Scaling.MX_BLOCK, 128), a, b, n, truth);

        System.out.println();
        System.out.println("mxfp4, rounding of the cast varied, fp32 sequential");
        System.out.println("  policy                       relative error");
        NumericPolicy m4 = NumericPolicy.mxfp4();
        row("nearest even", m4, a, b, n, truth);
        row("stochastic, seed 1", m4.withRounding("sr", Rounding.STOCHASTIC,
                Rounding.NEAREST_EVEN), a, b, n, truth, 1L);
        row("stochastic, seed 2", m4.withRounding("sr", Rounding.STOCHASTIC,
                Rounding.NEAREST_EVEN), a, b, n, truth, 2L);
    }

    private static void row(String label, NumericPolicy p,
                            double[] a, double[] b, int n, double truth) {
        row(label, p, a, b, n, truth, 12345L);
    }

    private static void row(String label, NumericPolicy p,
                            double[] a, double[] b, int n, double truth, long seed) {
        Rng rng = new Rng(seed);
        double[] qa = new double[n];
        double[] qb = new double[n];

        double scaleA = p.inputFormat() == null ? 1.0
                : Quant.perTensorScale(p.inputFormat(), Quant.amax(a, 0, n));
        double scaleB = p.inputFormat() == null ? 1.0
                : Quant.perTensorScale(p.inputFormat(), Quant.amax(b, 0, n));

        Quant.quantize(a, 0, qa, 0, n, p, scaleA, rng);
        Quant.quantize(b, 0, qb, 0, n, p, scaleB, rng);

        double got = Dot.dot(qa, 0, 1, qb, 0, 1, n, p, rng);
        double rel = Math.abs(got - truth) / Math.abs(truth);
        System.out.printf(Locale.ROOT, "  %-28s %.3e%n", label, rel);
    }

    private OrderDemo() {
    }
}
