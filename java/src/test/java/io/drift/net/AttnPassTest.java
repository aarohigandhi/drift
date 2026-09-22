package io.drift.net;

import io.drift.kernel.NumericPolicy;
import io.drift.train.AttnNet;
import io.drift.train.AttnPass;
import io.drift.train.Grads;
import io.drift.train.ModelPass;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The exact pass is the reference every policy is measured against, so the attention
 * gradients are checked against central finite differences on every parameter.
 */
class AttnPassTest {

    private static final int VOCAB = 7;
    private static final int CTX = 4;
    private static final int DIM = 6;
    private static final int HIDDEN = 5;
    private static final int BATCH = 2;

    private final int[] contexts = {0, 1, 2, 3, 4, 5, 6, 0};
    private final int[] targets = {1, 2, 3, 4, 5, 6, 0, 1};

    @Test
    void exactGradientsMatchFiniteDifferences() {
        AttnNet net = new AttnNet(VOCAB, CTX, DIM, HIDDEN, 31);
        ModelPass pass = net.pass(NumericPolicy.exact(), BATCH, 1);
        Grads g = new Grads(net);
        pass.lossAndGrads(contexts, targets, g);

        double[][] params = net.tensors();
        double[][] grads = g.tensors();
        double step = 1e-6;
        double worst = 0;
        for (int t = 0; t < params.length; t++) {
            for (int i = 0; i < params[t].length; i++) {
                double keep = params[t][i];
                params[t][i] = keep + step;
                double up = pass.forward(contexts, targets);
                params[t][i] = keep - step;
                double down = pass.forward(contexts, targets);
                params[t][i] = keep;
                double numeric = (up - down) / (2 * step);
                double err = Math.abs(numeric - grads[t][i])
                        / Math.max(1e-6, Math.abs(numeric) + Math.abs(grads[t][i]));
                worst = Math.max(worst, err);
            }
        }
        assertTrue(worst < 1e-5, "worst relative gradient error " + worst);
    }

    /** Causal masking: a position must not attend to anything after it. */
    @Test
    void attentionIsCausalAndNormalised() {
        AttnNet net = new AttnNet(VOCAB, CTX, DIM, HIDDEN, 5);
        AttnPass pass = (AttnPass) net.pass(NumericPolicy.exact(), BATCH, 1);
        pass.forward(contexts, targets);
        double[] prob = pass.attentionProbabilities();
        for (int b = 0; b < BATCH; b++) {
            for (int i = 0; i < CTX; i++) {
                double sum = 0;
                for (int s = 0; s < CTX; s++) {
                    double a = prob[b * CTX * CTX + i * CTX + s];
                    if (s > i) {
                        assertEquals(0.0, a, "position " + i + " attended to later position " + s);
                    }
                    sum += a;
                }
                assertEquals(1.0, sum, 1e-12, "row " + i + " of batch " + b);
            }
        }
    }
}
