package io.drift.train;

/** Gradient buffers with the same shapes and order as {@link Model#tensors()}. */
public final class Grads {

    private final double[][] tensors;

    public Grads(Model m) {
        double[][] params = m.tensors();
        tensors = new double[params.length][];
        for (int i = 0; i < params.length; i++) {
            tensors[i] = new double[params[i].length];
        }
    }

    public double[][] tensors() {
        return tensors;
    }

    public double[] get(int index) {
        return tensors[index];
    }

    public void zero() {
        for (double[] t : tensors) {
            java.util.Arrays.fill(t, 0.0);
        }
    }
}
