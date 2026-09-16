package io.drift.train;

import io.drift.kernel.NumericPolicy;

/** One model's forward and backward pass under one policy, with its own workspace. */
public interface ModelPass {

    NumericPolicy policy();

    /** Mean cross-entropy over every target in the batch, with gradients written into {@code g}. */
    double lossAndGrads(int[] contexts, int[] targets, Grads g);

    /** Mean cross-entropy only. */
    double forward(int[] contexts, int[] targets);

    /**
     * Cumulative clamp counts: clamped and cast activations, clamped and cast weights,
     * clamped and cast gradients.
     */
    long[] clampCounts();
}
