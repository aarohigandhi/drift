package io.drift.train;

import io.drift.kernel.NumericPolicy;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A model the trainer can run: parameters as named tensors, and a way to make a pass
 * that computes loss and gradients under a numeric policy.
 */
public interface Model {

    /** Parameter tensors in a fixed order. Gradients use the same order. */
    double[][] tensors();

    String[] tensorNames();

    /** The weight matrices that enter a policy-governed matmul, for per-layer reporting. */
    String[] matmulNames();

    int vocab();

    /** Characters of context per window. */
    int contextLength();

    /**
     * Targets per window: 1 when only the character after the window is predicted, or
     * {@code contextLength()} when every position predicts its successor.
     */
    int targetsPerWindow();

    ModelPass pass(NumericPolicy p, int batch, long rngSeed);

    void save(Path path) throws IOException;

    default int paramCount() {
        int n = 0;
        for (double[] t : tensors()) {
            n += t.length;
        }
        return n;
    }

    /** Round every parameter to FP32, which is what a run with FP32 master weights stores. */
    default void roundToFp32() {
        for (double[] t : tensors()) {
            for (int i = 0; i < t.length; i++) {
                t[i] = (float) t[i];
            }
        }
    }

    default int indexOf(String tensorName) {
        String[] names = tensorNames();
        for (int i = 0; i < names.length; i++) {
            if (names[i].equals(tensorName)) {
                return i;
            }
        }
        throw new IllegalArgumentException("no tensor " + tensorName);
    }
}
