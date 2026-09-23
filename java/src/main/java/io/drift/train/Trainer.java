package io.drift.train;

import io.drift.kernel.NumericPolicy;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

/**
 * One training run under one policy, with the exact reference running alongside.
 *
 * <p>Every run with the same seed starts from the same weights and sees the same
 * batches in the same order, so two runs differ only in their policy.
 *
 * <p>The instrument. Every {@code measureEvery} steps the gradient is computed a
 * second time, from the same weights and the same batch, with nothing cast and
 * everything in FP64. Comparing the two separates what the arithmetic did to this
 * step from where the run has drifted to, which a loss curve cannot do:
 *
 * <ul>
 *   <li>{@code cos}: cosine between the policy gradient and the exact one. Direction.
 *   <li>{@code gain}: projection of the policy gradient onto the exact one, divided
 *       by the exact norm squared. 1 is unbiased along the true direction; below 1
 *       the arithmetic is shrinking the step, above 1 inflating it.
 *   <li>{@code rel}: norm of the difference over the exact norm. Total error.
 * </ul>
 */
public final class Trainer {

    public static final class Config {
        public int ctx = 8;
        public int emb = 16;
        public int hidden = 128;
        public int batch = 32;
        public int steps = 3000;
        public double lr = 2e-3;
        public int logEvery = 10;
        public int measureEvery = 25;
        public int evalEvery = 250;
        public int evalBatches = 32;
        /**
         * Policies whose gradient is also computed at every measurement, from this run's
         * weights and batch. They never touch the update, so the run is unchanged; they
         * answer what a different arithmetic would have done at the same point.
         */
        public String[] probes = {};
        /** Write the final weights next to the run's log, for tools that need real tensors. */
        public boolean saveWeights = false;

        /** Which model to train: "mlp" or "attn". */
        public String model = "mlp";

        Model buildModel(int vocab, long seed) {
            return switch (model) {
                case "mlp" -> new Net(vocab, ctx, emb, hidden, seed);
                case "attn" -> new AttnNet(vocab, ctx, emb, hidden, seed);
                default -> throw new IllegalArgumentException("unknown model " + model);
            };
        }
    }

    private final Config cfg;
    private final Corpus corpus;

    public Trainer(Config cfg, Corpus corpus) {
        this.cfg = cfg;
        this.corpus = corpus;
    }

    public record Summary(String policy, long seed, int steps, boolean diverged,
                          double finalTrainLoss, double finalValLoss, double meanCos,
                          double meanGain, double meanRel, double seconds) {
    }

    public Summary run(NumericPolicy policy, long seed, Path out) throws IOException {
        long started = System.nanoTime();
        Model net = cfg.buildModel(corpus.vocab, 1000 + seed);
        if (policy.masterFp32) {
            net.roundToFp32();
        }

        ModelPass pass = net.pass(policy, cfg.batch, 0xC0FFEEL * (seed + 1));
        ModelPass exact = net.pass(NumericPolicy.exact(), cfg.batch, 1);
        Grads g = new Grads(net);
        Grads ge = new Grads(net);

        ModelPass[] probes = new ModelPass[cfg.probes.length];
        for (int i = 0; i < probes.length; i++) {
            probes[i] = net.pass(Policies.byName(cfg.probes[i]), cfg.batch, 0xBEEFL * (seed + 1) + i);
        }
        Grads gp = probes.length > 0 ? new Grads(net) : null;

        double[][] params = net.tensors();
        double[][] grads = g.tensors();
        double[][] m = new double[params.length][];
        double[][] v = new double[params.length][];
        for (int t = 0; t < params.length; t++) {
            m[t] = new double[params[t].length];
            v[t] = new double[params[t].length];
        }

        int[] ctxBuf = new int[cfg.batch * cfg.ctx];
        int[] tgtBuf = new int[cfg.batch * net.targetsPerWindow()];
        Random batches = new Random(seed);   // same batch stream for every policy

        double initialLoss = Math.log(corpus.vocab);
        boolean diverged = false;
        double lastLoss = Double.NaN;
        double lastVal = Double.NaN;
        double cosSum = 0;
        double gainSum = 0;
        double relSum = 0;
        int measured = 0;
        int step = 0;

        Files.createDirectories(out.getParent());
        try (BufferedWriter w = Files.newBufferedWriter(out)) {
            w.write(String.format(Locale.ROOT,
                    "{\"kind\":\"config\",\"policy\":\"%s\",\"seed\":%d,\"describe\":\"%s\","
                            + "\"model\":\"%s\",\"params\":%d,\"vocab\":%d,\"ctx\":%d,\"emb\":%d,\"hidden\":%d,"
                            + "\"batch\":%d,\"steps\":%d,\"lr\":%s}%n",
                    policy.name, seed, policy.describe(), cfg.model, net.paramCount(), corpus.vocab,
                    cfg.ctx, cfg.emb, cfg.hidden, cfg.batch, cfg.steps, cfg.lr));

            for (step = 1; step <= cfg.steps; step++) {
                sample(corpus.train, batches, ctxBuf, tgtBuf, net.targetsPerWindow());
                double loss = pass.lossAndGrads(ctxBuf, tgtBuf, g);
                lastLoss = loss;

                if (!Double.isFinite(loss) || loss > 2 * initialLoss) {
                    diverged = true;
                    w.write(String.format(Locale.ROOT,
                            "{\"kind\":\"diverged\",\"step\":%d,\"loss\":%s}%n", step, json(loss)));
                    break;
                }

                if (step % cfg.measureEvery == 0 || step == 1) {
                    double lossExact = exact.lossAndGrads(ctxBuf, tgtBuf, ge);
                    String layers = compareLayers(net, g, ge);
                    double[] all = compare(grads, ge.tensors());
                    cosSum += all[0];
                    gainSum += all[1];
                    relSum += all[2];
                    measured++;
                    StringBuilder probeJson = new StringBuilder();
                    for (int i = 0; i < probes.length; i++) {
                        probes[i].lossAndGrads(ctxBuf, tgtBuf, gp);
                        double[] c = compare(gp.tensors(), ge.tensors());
                        if (i > 0) {
                            probeJson.append(',');
                        }
                        probeJson.append(String.format(Locale.ROOT,
                                "\"%s\":{\"cos\":%s,\"gain\":%s,\"rel\":%s,\"layers\":{%s}}",
                                probes[i].policy().name, json(c[0]), json(c[1]), json(c[2]), compareLayers(net, gp, ge)));
                    }
                    long[] cc = pass.clampCounts();
                    w.write(String.format(Locale.ROOT,
                            "{\"kind\":\"measure\",\"step\":%d,\"loss\":%s,\"loss_exact\":%s,"
                                    + "\"cos\":%s,\"gain\":%s,\"rel\":%s,\"layers\":{%s},"
                                    + "\"clamp_x\":%s,\"clamp_w\":%s,\"clamp_g\":%s,"
                                    + "\"clamp_prob\":%s,\"underflow\":%s,\"spread_binades\":%s,"
                                    + "\"cast_rel_err\":%s,\"block_spread\":%s,\"cast_bias\":%s,"
                                    + "\"probes\":{%s}}%n",
                            step, json(loss), json(lossExact), json(all[0]), json(all[1]), json(all[2]),
                            layers, json(frac(cc[0], cc[1])), json(frac(cc[2], cc[3])), json(frac(cc[4], cc[5])),
                            json(cc.length > 6 ? frac(cc[6], cc[7]) : 0.0),
                            json(cc.length > 8 ? frac(cc[8], cc[9]) : 0.0),
                            json(cc.length > 10 ? (double) cc[10] : 0.0),
                            json(cc.length > 12 ? Double.longBitsToDouble(cc[12]) : 0.0),
                            json(cc.length > 13 ? Double.longBitsToDouble(cc[13]) : 0.0),
                            json(cc.length > 14 ? Double.longBitsToDouble(cc[14]) : 0.0), probeJson));
                }

                adam(params, grads, m, v, step, policy.masterFp32);

                if (step % cfg.logEvery == 0) {
                    w.write(String.format(Locale.ROOT,
                            "{\"kind\":\"train\",\"step\":%d,\"loss\":%s}%n", step, json(loss)));
                }
                if (step % cfg.evalEvery == 0 || step == cfg.steps) {
                    lastVal = evaluate(exact, net.targetsPerWindow());
                    w.write(String.format(Locale.ROOT,
                            "{\"kind\":\"eval\",\"step\":%d,\"val_loss\":%s}%n", step, json(lastVal)));
                }
            }

            if (cfg.saveWeights && !diverged) {
                net.save(out.resolveSibling(out.getFileName().toString().replace(".jsonl", ".weights")));
            }

            double secs = (System.nanoTime() - started) / 1e9;
            Summary s = new Summary(policy.name, seed, Math.min(step, cfg.steps), diverged, lastLoss,
                    lastVal, cosSum / Math.max(1, measured), gainSum / Math.max(1, measured),
                    relSum / Math.max(1, measured), secs);
            w.write(String.format(Locale.ROOT,
                    "{\"kind\":\"summary\",\"diverged\":%b,\"train_loss\":%s,\"val_loss\":%s,"
                            + "\"mean_cos\":%s,\"mean_gain\":%s,\"mean_rel\":%s,\"seconds\":%.1f}%n",
                    diverged, json(lastLoss), json(lastVal), json(s.meanCos()), json(s.meanGain()),
                    json(s.meanRel()), secs));
            return s;
        }
    }

    /**
     * Held-out loss, always computed exactly, so it measures the weights a policy
     * produced rather than the policy's arithmetic at evaluation time.
     */
    private double evaluate(ModelPass exact, int targetsPerWindow) {
        Random r = new Random(424242);   // same held-out windows every time
        int[] ctxBuf = new int[cfg.batch * cfg.ctx];
        int[] tgtBuf = new int[cfg.batch * targetsPerWindow];
        double sum = 0;
        for (int b = 0; b < cfg.evalBatches; b++) {
            sample(corpus.val, r, ctxBuf, tgtBuf, targetsPerWindow);
            sum += exact.forward(ctxBuf, tgtBuf);
        }
        return sum / cfg.evalBatches;
    }

    /**
     * One random window per batch row. With one target per window the target is the
     * character after it; with one per position, each position predicts the character
     * after it. The random draws are identical either way, so adding the second case
     * leaves the MLP's batch stream unchanged.
     */
    private void sample(int[] tokens, Random r, int[] ctxBuf, int[] tgtBuf, int targetsPerWindow) {
        for (int i = 0; i < cfg.batch; i++) {
            int start = r.nextInt(tokens.length - cfg.ctx - 1);
            System.arraycopy(tokens, start, ctxBuf, i * cfg.ctx, cfg.ctx);
            if (targetsPerWindow == 1) {
                tgtBuf[i] = tokens[start + cfg.ctx];
            } else {
                System.arraycopy(tokens, start + 1, tgtBuf, i * cfg.ctx, cfg.ctx);
            }
        }
    }

    private static final double B1 = 0.9;
    private static final double B2 = 0.999;
    private static final double EPS = 1e-8;

    /** Adam. With FP32 masters, the weights and both moments are stored as floats. */
    private void adam(double[][] params, double[][] grads, double[][] m, double[][] v,
                      int step, boolean fp32) {
        double c1 = 1 - Math.pow(B1, step);
        double c2 = 1 - Math.pow(B2, step);
        for (int t = 0; t < params.length; t++) {
            double[] p = params[t];
            double[] g = grads[t];
            double[] mt = m[t];
            double[] vt = v[t];
            for (int i = 0; i < p.length; i++) {
                double mi = B1 * mt[i] + (1 - B1) * g[i];
                double vi = B2 * vt[i] + (1 - B2) * g[i] * g[i];
                double pi = p[i] - cfg.lr * (mi / c1) / (Math.sqrt(vi / c2) + EPS);
                if (fp32) {
                    mi = (float) mi;
                    vi = (float) vi;
                    pi = (float) pi;
                }
                mt[i] = mi;
                vt[i] = vi;
                p[i] = pi;
            }
        }
    }

    /** cos, gain and rel over all tensors concatenated. */
    static double[] compare(double[][] got, double[][] exact) {
        double dot = 0;
        double gg = 0;
        double ee = 0;
        double dd = 0;
        for (int t = 0; t < got.length; t++) {
            for (int i = 0; i < got[t].length; i++) {
                double a = got[t][i];
                double b = exact[t][i];
                dot += a * b;
                gg += a * a;
                ee += b * b;
                dd += (a - b) * (a - b);
            }
        }
        return new double[]{dot / Math.sqrt(gg * ee), dot / ee, Math.sqrt(dd / ee)};
    }

    private static String compareLayers(Model model, Grads g, Grads ge) {
        String[] names = model.matmulNames();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++) {
            int t = model.indexOf(names[i]);
            double[] c = compare(new double[][]{g.get(t)}, new double[][]{ge.get(t)});
            if (i > 0) {
                sb.append(',');
            }
            sb.append(String.format(Locale.ROOT, "\"%s\":{\"cos\":%s,\"gain\":%s,\"rel\":%s}",
                    names[i], json(c[0]), json(c[1]), json(c[2])));
        }
        return sb.toString();
    }

    private static double frac(long a, long b) {
        return b == 0 ? 0.0 : (double) a / b;
    }

    private static String json(double x) {
        return Double.isFinite(x) ? String.format(Locale.ROOT, "%.8g", x) : "null";
    }
}
