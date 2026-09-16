package io.drift.train;

import java.util.Random;

/**
 * A character-level MLP language model: embed the last {@code ctx} characters,
 * concatenate, two tanh layers, softmax over the vocabulary.
 *
 * <p>Deliberately not a transformer. Everything this project varies lives inside the
 * matmul, and an MLP of this size is three matmuls forward and six back, with no
 * attention to add numerical behaviour of its own that would then need separating
 * out. The limitation is stated in the README rather than hidden: what is measured
 * here is what low precision does to matrix multiplication inside a real training
 * loop on real text, not what it does to attention.
 */
public final class Net implements Model {

    public final int vocab;
    public final int ctx;
    public final int emb;
    public final int hidden;
    public final int in1;

    public final double[] embed;
    public final double[] w1;
    public final double[] b1;
    public final double[] w2;
    public final double[] b2;
    public final double[] w3;
    public final double[] b3;

    public Net(int vocab, int ctx, int emb, int hidden, long seed) {
        this.vocab = vocab;
        this.ctx = ctx;
        this.emb = emb;
        this.hidden = hidden;
        this.in1 = ctx * emb;

        embed = new double[vocab * emb];
        w1 = new double[hidden * in1];
        b1 = new double[hidden];
        w2 = new double[hidden * hidden];
        b2 = new double[hidden];
        w3 = new double[vocab * hidden];
        b3 = new double[vocab];

        Random r = new Random(seed);
        gaussian(embed, 1.0, r);
        gaussian(w1, 1.0 / Math.sqrt(in1), r);
        gaussian(w2, 1.0 / Math.sqrt(hidden), r);
        gaussian(w3, 1.0 / Math.sqrt(hidden), r);
    }

    /** Parameter tensors in a fixed order, shared with {@link Grads#tensors()}. */
    public double[][] tensors() {
        return new double[][]{embed, w1, b1, w2, b2, w3, b3};
    }

    public static final String[] TENSOR_NAMES = {"embed", "w1", "b1", "w2", "b2", "w3", "b3"};

    @Override
    public String[] tensorNames() {
        return TENSOR_NAMES;
    }

    @Override
    public String[] matmulNames() {
        return new String[]{"w1", "w2", "w3"};
    }

    @Override
    public int vocab() {
        return vocab;
    }

    @Override
    public int contextLength() {
        return ctx;
    }

    @Override
    public int targetsPerWindow() {
        return 1;
    }

    @Override
    public ModelPass pass(io.drift.kernel.NumericPolicy p, int batch, long rngSeed) {
        return new Pass(p, this, batch, rngSeed);
    }

    /** Round every parameter to FP32, which is what a run with FP32 master weights stores. */
    @Override
    public void roundToFp32() {
        for (double[] t : tensors()) {
            for (int i = 0; i < t.length; i++) {
                t[i] = (float) t[i];
            }
        }
    }

    @Override
    public int paramCount() {
        int n = 0;
        for (double[] t : tensors()) {
            n += t.length;
        }
        return n;
    }

    /** Dimensions, then every tensor, as raw doubles. Exact, so a reload is bit-identical. */
    @Override
    public void save(java.nio.file.Path path) throws java.io.IOException {
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(vocab);
            out.writeInt(ctx);
            out.writeInt(emb);
            out.writeInt(hidden);
            for (double[] t : tensors()) {
                out.writeInt(t.length);
                for (double v : t) {
                    out.writeDouble(v);
                }
            }
        }
    }

    public static Net load(java.nio.file.Path path) throws java.io.IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(path)))) {
            Net n = new Net(in.readInt(), in.readInt(), in.readInt(), in.readInt(), 0);
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
