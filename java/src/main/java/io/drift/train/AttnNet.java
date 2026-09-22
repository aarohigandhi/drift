package io.drift.train;

import io.drift.kernel.NumericPolicy;

import java.util.Random;

/**
 * One block of causal self attention followed by an MLP, with residual connections,
 * predicting the next character at every position.
 *
 * <p>This exists for the two things attention does that the {@link Net} MLP cannot.
 * A score is a product of two activations rather than an activation times a weight,
 * so both operands carry cast error. And a softmax turns those scores into weights
 * that sum to one, which are then themselves an operand of the next matmul, and
 * which crowd just below 1.0, the same place that makes a shared exponent clamp.
 *
 * <p>There is no layer normalisation. Fewer moving parts means a difference between
 * two policies is easier to attribute, and at this size the model trains without it.
 * Single head, for the same reason.
 */
public final class AttnNet implements Model {

    public final int vocab;
    public final int ctx;
    public final int dim;
    public final int hidden;

    public final double[] embed;
    public final double[] pos;
    public final double[] wq;
    public final double[] wk;
    public final double[] wv;
    public final double[] wo;
    public final double[] w1;
    public final double[] b1;
    public final double[] w2;
    public final double[] b2;
    public final double[] wout;
    public final double[] bout;

    public static final String[] TENSOR_NAMES = {
            "embed", "pos", "wq", "wk", "wv", "wo", "w1", "b1", "w2", "b2", "wout", "bout"};

    public AttnNet(int vocab, int ctx, int dim, int hidden, long seed) {
        this.vocab = vocab;
        this.ctx = ctx;
        this.dim = dim;
        this.hidden = hidden;

        embed = new double[vocab * dim];
        pos = new double[ctx * dim];
        wq = new double[dim * dim];
        wk = new double[dim * dim];
        wv = new double[dim * dim];
        wo = new double[dim * dim];
        w1 = new double[hidden * dim];
        b1 = new double[hidden];
        w2 = new double[dim * hidden];
        b2 = new double[dim];
        wout = new double[vocab * dim];
        bout = new double[vocab];

        Random r = new Random(seed);
        gaussian(embed, 1.0 / Math.sqrt(dim), r);
        gaussian(pos, 0.02, r);
        double s = 1.0 / Math.sqrt(dim);
        gaussian(wq, s, r);
        gaussian(wk, s, r);
        gaussian(wv, s, r);
        gaussian(wo, s, r);
        gaussian(w1, s, r);
        gaussian(w2, 1.0 / Math.sqrt(hidden), r);
        gaussian(wout, s, r);
    }

    @Override
    public double[][] tensors() {
        return new double[][]{embed, pos, wq, wk, wv, wo, w1, b1, w2, b2, wout, bout};
    }

    @Override
    public String[] tensorNames() {
        return TENSOR_NAMES;
    }

    @Override
    public String[] matmulNames() {
        return new String[]{"wq", "wk", "wv", "wo", "w1", "w2", "wout"};
    }

    @Override
    public int vocab() {
        return vocab;
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    /** Every position predicts the character after it. */
    @Override
    public int targetsPerWindow() {
        return ctx;
    }

    @Override
    public ModelPass pass(NumericPolicy p, int batch, long rngSeed) {
        return new AttnPass(p, this, batch, rngSeed);
    }

    @Override
    public void save(java.nio.file.Path path) throws java.io.IOException {
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(vocab);
            out.writeInt(ctx);
            out.writeInt(dim);
            out.writeInt(hidden);
            for (double[] t : tensors()) {
                out.writeInt(t.length);
                for (double v : t) {
                    out.writeDouble(v);
                }
            }
        }
    }

    public static AttnNet load(java.nio.file.Path path) throws java.io.IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(path)))) {
            AttnNet n = new AttnNet(in.readInt(), in.readInt(), in.readInt(), in.readInt(), 0);
            for (double[] t : n.tensors()) {
                int len = in.readInt();
                if (len != t.length) {
                    throw new java.io.IOException("tensor length " + len + ", expected " + t.length);
                }
                for (int i = 0; i < len; i++) {
                    t[i] = in.readDouble();
                }
            }
            return n;
        }
    }

    private static void gaussian(double[] a, double std, Random r) {
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextGaussian() * std;
        }
    }
}
