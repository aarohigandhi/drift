package io.drift.net;

import io.drift.kernel.NumericPolicy;
import io.drift.train.Grads;
import io.drift.train.Net;
import io.drift.train.Pass;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The exact pass is the reference every policy is measured against, so its gradients
 * are checked against central finite differences on every parameter of a small net.
 */
class PassTest {

    private static final int VOCAB = 7;
    private static final int CTX = 3;
    private static final int EMB = 4;
    private static final int HIDDEN = 5;
    private static final int BATCH = 4;

    private final int[] contexts = {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 2};
    private final int[] targets = {3, 6, 1, 0};

    @Test
    void exactGradientsMatchFiniteDifferences() {
        Net net = new Net(VOCAB, CTX, EMB, HIDDEN, 17);
        Pass pass = new Pass(NumericPolicy.exact(), net, BATCH, 1);
        Grads g = new Grads(net);
        pass.lossAndGrads(contexts, targets, g);

        double[][] params = net.tensors();
        double[][] grads = g.tensors();
        double h = 1e-6;
        double worst = 0;
        for (int t = 0; t < params.length; t++) {
            for (int i = 0; i < params[t].length; i++) {
                double keep = params[t][i];
                params[t][i] = keep + h;
                double up = pass.forward(contexts, targets);
                params[t][i] = keep - h;
                double down = pass.forward(contexts, targets);
                params[t][i] = keep;
                double numeric = (up - down) / (2 * h);
                double err = Math.abs(numeric - grads[t][i]) / Math.max(1e-6, Math.abs(numeric) + Math.abs(grads[t][i]));
                worst = Math.max(worst, err);
            }
        }
        assertTrue(worst < 1e-5, "worst relative gradient error " + worst);
    }

    @Test
    void savedWeightsReloadBitIdentical(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws java.io.IOException {
        Net net = new Net(VOCAB, CTX, EMB, HIDDEN, 23);
        java.nio.file.Path file = dir.resolve("net.weights");
        net.save(file);
        Net back = Net.load(file);
        double[][] a = net.tensors();
        double[][] b = back.tensors();
        for (int t = 0; t < a.length; t++) {
            for (int i = 0; i < a[t].length; i++) {
                assertTrue(Double.doubleToRawLongBits(a[t][i]) == Double.doubleToRawLongBits(b[t][i]),
                        Net.TENSOR_NAMES[t] + "[" + i + "]");
            }
        }
    }

    @Test
    void uncastFp32PassStaysCloseToExact() {
        Net net = new Net(VOCAB, CTX, EMB, HIDDEN, 17);
        Grads ge = new Grads(net);
        Grads gf = new Grads(net);
        new Pass(NumericPolicy.exact(), net, BATCH, 1).lossAndGrads(contexts, targets, ge);
        new Pass(NumericPolicy.builder("fp32").build(), net, BATCH, 1).lossAndGrads(contexts, targets, gf);

        double dot = 0;
        double ee = 0;
        double ff = 0;
        double[][] a = gf.tensors();
        double[][] b = ge.tensors();
        for (int t = 0; t < a.length; t++) {
            for (int i = 0; i < a[t].length; i++) {
                dot += a[t][i] * b[t][i];
                ee += b[t][i] * b[t][i];
                ff += a[t][i] * a[t][i];
            }
        }
        assertTrue(dot / Math.sqrt(ee * ff) > 1 - 1e-9);
    }
}
