package io.drift.tools;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.Dot;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Acc;
import io.drift.kernel.NumericPolicy.Order;
import io.drift.kernel.NumericPolicy.Scaling;
import io.drift.kernel.Quant;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Inner products under many policies, many seeds and several lengths.
 *
 * <p>Every policy sees the same vector pairs, so differences between rows are the
 * policy and nothing else. Vectors are a normal draw times a lognormal scale, which
 * gives the heavy tail and the few binades of spread that activations have.
 */
public final class DotSweep {

    private static final int[] LENGTHS = {256, 1024, 4096};
    private static final int SEEDS = 200;

    record Case(String group, String label, NumericPolicy policy) {
    }

    public static void main(String[] args) throws IOException {
        Path out = Path.of(args.length > 0 ? args[0] : "../results/dot_sweep.csv");
        Files.createDirectories(out.toAbsolutePath().getParent());

        List<Case> cases = cases();
        try (BufferedWriter w = Files.newBufferedWriter(out)) {
            w.write("group,label,n,seed,rel_error,signed_error,clamped,result,exact\n");
            for (int n : LENGTHS) {
                for (int seed = 0; seed < SEEDS; seed++) {
                    double[] a = new double[n];
                    double[] b = new double[n];
                    fill(a, b, seed * 7919L + n);
                    double exact = Dot.dot(a, 0, 1, b, 0, 1, n, NumericPolicy.exact(), null);
                    for (Case c : cases) {
                        Result r = run(c.policy, a, b, n, seed);
                        double rel = Math.abs(r.value - exact) / Math.abs(exact);
                        w.write(String.format(Locale.ROOT, "%s,%s,%d,%d,%.6e,%.6e,%d,%.17g,%.17g%n",
                                c.group, c.label, n, seed, rel, (r.value - exact) / Math.abs(exact),
                                r.clamped, r.value, exact));
                    }
                }
            }
        }

        summarise(out, cases);
    }

    static List<Case> cases() {
        List<Case> cs = new ArrayList<>();

        // Accumulation alone: nothing is cast, so every bit of error is the summation.
        for (Acc acc : new Acc[]{Acc.FP32, Acc.FP16, Acc.BF16}) {
            for (Order o : Order.values()) {
                String label = acc.name().toLowerCase(Locale.ROOT) + " " + o.name().toLowerCase(Locale.ROOT);
                cs.add(new Case("accumulation", label,
                        NumericPolicy.builder(label).acc(acc).order(o).accBlock(32).build()));
            }
        }

        // The cast and the accumulator together: bf16 inputs, varied summation.
        for (Acc acc : new Acc[]{Acc.FP32, Acc.FP16, Acc.BF16}) {
            for (Order o : new Order[]{Order.SEQUENTIAL, Order.PAIRWISE}) {
                String label = "bf16 in / " + acc.name().toLowerCase(Locale.ROOT) + " " + o.name().toLowerCase(Locale.ROOT);
                cs.add(new Case("cast+accumulation", label,
                        NumericPolicy.builder(label).input(Formats.BF16).acc(acc).order(o).build()));
            }
        }

        // Scaling, for an 8 bit and a 4 bit element format.
        for (MiniFloat f : new MiniFloat[]{Formats.E4M3, Formats.E2M1}) {
            String fn = f.name;
            cs.add(new Case("scaling", fn + " none",
                    NumericPolicy.builder(fn + " none").input(f).scaling(Scaling.NONE).build()));
            cs.add(new Case("scaling", fn + " per-tensor",
                    NumericPolicy.builder(fn + " per-tensor").input(f).scaling(Scaling.PER_TENSOR).build()));
            cs.add(new Case("scaling", fn + " mx32 spec",
                    NumericPolicy.builder(fn + " mx32 spec").input(f).scaling(Scaling.MX_BLOCK).build()));
            cs.add(new Case("scaling", fn + " mx32 +1",
                    NumericPolicy.builder(fn + " mx32 +1").input(f).scaling(Scaling.MX_BLOCK).mxHeadroom(1).build()));
        }

        // Rounding of a 4 bit cast.
        cs.add(new Case("rounding", "e2m1 mx32 nearest",
                NumericPolicy.builder("n").input(Formats.E2M1).scaling(Scaling.MX_BLOCK).build()));
        cs.add(new Case("rounding", "e2m1 mx32 stochastic",
                NumericPolicy.builder("s").input(Formats.E2M1).scaling(Scaling.MX_BLOCK)
                        .inputRounding(Rounding.STOCHASTIC).build()));
        return cs;
    }

    record Result(double value, int clamped) {
    }

    static Result run(NumericPolicy p, double[] a, double[] b, int n, int seed) {
        Rng rng = new Rng(0xD1F7L + seed);
        double[] qa = new double[n];
        double[] qb = new double[n];
        MiniFloat f = p.inputFormat;
        double sa = f == null ? 1.0 : Quant.perTensorScale(f, Quant.amax(a, 0, n));
        double sb = f == null ? 1.0 : Quant.perTensorScale(f, Quant.amax(b, 0, n));
        int clamped = Quant.quantize(a, 0, qa, 0, n, f, p, sa, rng)
                + Quant.quantize(b, 0, qb, 0, n, f, p, sb, rng);
        return new Result(Dot.dot(qa, 0, 1, qb, 0, 1, n, p, rng), clamped);
    }

    static void fill(double[] a, double[] b, long seed) {
        Random r = new Random(seed);
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
            b[i] = r.nextGaussian() * Math.exp(r.nextGaussian() * 0.8);
        }
    }

    /** Median relative error per policy and length, straight from the file just written. */
    private static void summarise(Path csv, List<Case> cases) throws IOException {
        List<String> lines = Files.readAllLines(csv);
        System.out.printf(Locale.ROOT, "%-18s %-26s %12s %12s %12s   clamped/pair@4096%n",
                "group", "policy", "n=256", "n=1024", "n=4096");
        for (Case c : cases) {
            StringBuilder sb = new StringBuilder();
            double clampedMean = 0;
            for (int n : LENGTHS) {
                double[] errs = new double[SEEDS];
                int k = 0;
                long clampSum = 0;
                for (String line : lines) {
                    String[] f = line.split(",");
                    if (f.length < 9 || !f[1].equals(c.label) || !f[2].equals(Integer.toString(n))) {
                        continue;
                    }
                    errs[k++] = Double.parseDouble(f[4]);
                    clampSum += Long.parseLong(f[6]);
                }
                Arrays.sort(errs, 0, k);
                sb.append(String.format(Locale.ROOT, " %12.3e", errs[k / 2]));
                if (n == 4096) {
                    clampedMean = (double) clampSum / k;
                }
            }
            System.out.printf(Locale.ROOT, "%-18s %-26s%s   %.1f%n", c.group, c.label, sb, clampedMean);
        }
    }

    private DotSweep() {
    }
}
