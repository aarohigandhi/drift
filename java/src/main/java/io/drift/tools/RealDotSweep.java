package io.drift.tools;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.kernel.Dot;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Acc;
import io.drift.kernel.NumericPolicy.Order;
import io.drift.kernel.NumericPolicy.Scaling;
import io.drift.kernel.Quant;
import io.drift.train.Corpus;
import io.drift.train.Net;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * The dot-product sweep again, on real tensors instead of synthetic ones.
 *
 * <p>DotSweep draws its vectors from a lognormal-scaled normal. That is a guess at what
 * activations look like. This takes the wide model's first layer, which is where its
 * accumulator error enters, and computes every pre-activation of a held-out batch:
 * the real 2,048-wide input of each window against each real row of the weight matrix.
 * Every policy sees the same pairs.
 *
 * <p>Two error measures. The median relative error matches DotSweep. The RMS ratio,
 * RMS of the error over RMS of the exact pre-activations, is the one that matters to
 * the layer: a pre-activation that is nearly zero can have a huge relative error that
 * tanh never notices.
 *
 * <pre>
 * --weights ../results/runs-real/fp32-s1.weights | --init 1
 * --data ../data/corpus.txt  --out ../results/real_dot.csv  --stage trained
 * </pre>
 */
public final class RealDotSweep {

    private static final int WINDOWS = 64;

    public static void main(String[] args) throws IOException {
        String weights = null;
        long initSeed = -1;
        String data = "../data/corpus.txt";
        String out = "../results/real_dot.csv";
        String stage = "trained";
        int ctx = 32;
        int emb = 64;
        int hidden = 64;
        for (int i = 0; i < args.length; i += 2) {
            switch (args[i]) {
                case "--weights" -> weights = args[i + 1];
                case "--init" -> initSeed = Long.parseLong(args[i + 1]);
                case "--data" -> data = args[i + 1];
                case "--out" -> out = args[i + 1];
                case "--stage" -> stage = args[i + 1];
                default -> throw new IllegalArgumentException("unknown flag " + args[i]);
            }
        }

        Corpus corpus = Corpus.load(Path.of(data), 0.1);
        // Initial weights are rebuilt from the seed exactly as Trainer builds them.
        Net net = weights != null ? Net.load(Path.of(weights))
                : new Net(corpus.vocab, ctx, emb, hidden, 1000 + initSeed);

        int in = net.in1;
        int units = net.hidden;
        double[] x = new double[WINDOWS * in];
        Random r = new Random(424242);
        for (int i = 0; i < WINDOWS; i++) {
            int start = r.nextInt(corpus.val.length - net.ctx - 1);
            for (int c = 0; c < net.ctx; c++) {
                System.arraycopy(net.embed, corpus.val[start + c] * net.emb, x, i * in + c * net.emb, net.emb);
            }
        }
        double[] w = net.w1;

        int pairs = WINDOWS * units;
        double[] exact = new double[pairs];
        NumericPolicy ex = NumericPolicy.exact();
        for (int i = 0; i < WINDOWS; i++) {
            for (int o = 0; o < units; o++) {
                exact[i * units + o] = Dot.dot(x, i * in, 1, w, o * in, 1, in, ex, null);
            }
        }

        Map<String, NumericPolicy> policies = new LinkedHashMap<>();
        for (Acc acc : new Acc[]{Acc.FP32, Acc.FP16, Acc.BF16}) {
            for (Order o : new Order[]{Order.SEQUENTIAL, Order.PAIRWISE, Order.BLOCKED}) {
                String name = acc.name().toLowerCase(Locale.ROOT) + " " + o.name().toLowerCase(Locale.ROOT);
                policies.put(name, NumericPolicy.builder(name).acc(acc).order(o).accBlock(32).build());
            }
        }
        policies.put("bf16 cast", NumericPolicy.builder("bf16 cast").input(Formats.BF16).build());
        policies.put("e4m3 per-tensor cast", NumericPolicy.builder("e4m3")
                .input(Formats.E4M3).scaling(Scaling.PER_TENSOR).build());
        policies.put("e4m3 mx32 cast", NumericPolicy.builder("e4m3 mx")
                .input(Formats.E4M3).scaling(Scaling.MX_BLOCK).build());

        Path outPath = Path.of(out);
        boolean fresh = !Files.exists(outPath);
        try (BufferedWriter bw = Files.newBufferedWriter(outPath, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (fresh) {
                bw.write("stage,policy,terms,pairs,median_rel_error,rms_error_ratio\n");
            }
            System.out.printf(Locale.ROOT, "%s: %d pairs of %d terms, exact pre-activation rms %.4f%n",
                    stage, pairs, in, rms(exact));
            System.out.printf(Locale.ROOT, "  %-22s %14s %14s%n", "policy", "median rel.", "rms ratio");
            for (Map.Entry<String, NumericPolicy> e : policies.entrySet()) {
                NumericPolicy p = e.getValue();
                MiniFloat f = p.inputFormat;
                Rng rng = new Rng(7);
                double[] qx = new double[x.length];
                double[] qw = new double[w.length];
                double sx = f == null ? 1.0 : Quant.perTensorScale(f, Quant.amax(x, 0, x.length));
                double sw = f == null ? 1.0 : Quant.perTensorScale(f, Quant.amax(w, 0, w.length));
                Quant.quantize(x, 0, qx, 0, x.length, f, p, sx, rng);
                Quant.quantize(w, 0, qw, 0, w.length, f, p, sw, rng);

                double[] rel = new double[pairs];
                double[] err = new double[pairs];
                for (int i = 0; i < WINDOWS; i++) {
                    for (int o = 0; o < units; o++) {
                        int k = i * units + o;
                        double got = Dot.dot(qx, i * in, 1, qw, o * in, 1, in, p, rng);
                        err[k] = got - exact[k];
                        rel[k] = Math.abs(err[k]) / Math.abs(exact[k]);
                    }
                }
                Arrays.sort(rel);
                double median = 0.5 * (rel[pairs / 2 - 1] + rel[pairs / 2]);
                double ratio = rms(err) / rms(exact);
                System.out.printf(Locale.ROOT, "  %-22s %14.3e %14.3e%n", e.getKey(), median, ratio);
                bw.write(String.format(Locale.ROOT, "%s,%s,%d,%d,%.6e,%.6e%n", stage, e.getKey(), in, pairs, median, ratio));
            }
        }
    }

    private static double rms(double[] a) {
        double s = 0;
        for (double v : a) {
            s += v * v;
        }
        return Math.sqrt(s / a.length);
    }

    private RealDotSweep() {
    }
}
