package io.drift.train;

/** Gradient buffers with the same shapes and order as {@link Net#tensors()}. */
public final class Grads {

    public final double[] embed;
    public final double[] w1;
    public final double[] b1;
    public final double[] w2;
    public final double[] b2;
    public final double[] w3;
    public final double[] b3;

    public Grads(Net n) {
        embed = new double[n.embed.length];
        w1 = new double[n.w1.length];
        b1 = new double[n.b1.length];
        w2 = new double[n.w2.length];
        b2 = new double[n.b2.length];
        w3 = new double[n.w3.length];
        b3 = new double[n.b3.length];
    }

    public double[][] tensors() {
        return new double[][]{embed, w1, b1, w2, b2, w3, b3};
    }

    public void zero() {
        for (double[] t : tensors()) {
            java.util.Arrays.fill(t, 0.0);
        }
    }
}
