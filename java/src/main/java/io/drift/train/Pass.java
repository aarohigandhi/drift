package io.drift.train;

import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.kernel.Dot;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Scaling;
import io.drift.kernel.Quant;

/**
 * One forward and backward pass under a numeric policy, with its own workspace.
 *
 * <p>What the policy governs is exactly the three linear layers: the activations and
 * weights entering each forward matmul are cast to the input format, the output
 * gradient entering each backward matmul is cast to the gradient format, and every
 * matmul accumulates under the policy's width, order and rounding. The forward pass
 * keeps its cast inputs and the backward pass reuses them, as a real FP8 kernel does.
 *
 * <p>Everything else stays in double: the embedding lookup, the biases, tanh, the
 * softmax and the loss. That is the usual scope of an FP8 recipe, and it keeps each
 * policy's effect confined to the thing the policy describes.
 *
 * <p>A pass holds state across steps, because delayed scaling needs an amax history,
 * so the policy pass and the exact reference pass are separate objects.
 */
public final class Pass implements ModelPass {

    private static final int LAYERS = 3;
    private static final int T_X = 0;
    private static final int T_W = 1;
    private static final int T_G = 2;

    public final NumericPolicy p;
    private final Rng rng;
    private final Net net;
    private final int batch;

    private final double[] x0;
    private final double[] h1;
    private final double[] h2;
    private final double[] logits;
    private final double[] dLogits;
    private final double[] dh2;
    private final double[] dh1;
    private final double[] dx0;

    private final double[][] qX = new double[LAYERS][];
    private final double[][] qW = new double[LAYERS][];
    private final double[][] qG = new double[LAYERS][];

    /** Ring buffers of recent amax per layer and tensor, for delayed scaling. */
    private final double[][][] history;
    private final int[][] historyCount;

    /** Elements clamped at the format maximum, cumulative, per tensor kind. */
    public long clampedX;
    public long clampedW;
    public long clampedG;
    /** Elements cast, cumulative, per tensor kind. */
    public long castX;
    public long castW;
    public long castG;

    public Pass(NumericPolicy p, Net net, int batch, long rngSeed) {
        this.p = p;
        this.net = net;
        this.batch = batch;
        this.rng = new Rng(rngSeed);

        int h = net.hidden;
        x0 = new double[batch * net.in1];
        h1 = new double[batch * h];
        h2 = new double[batch * h];
        logits = new double[batch * net.vocab];
        dLogits = new double[batch * net.vocab];
        dh2 = new double[batch * h];
        dh1 = new double[batch * h];
        dx0 = new double[batch * net.in1];

        int[] ins = {net.in1, h, h};
        int[] outs = {h, h, net.vocab};
        for (int l = 0; l < LAYERS; l++) {
            qX[l] = new double[batch * ins[l]];
            qW[l] = new double[outs[l] * ins[l]];
            qG[l] = new double[batch * outs[l]];
        }
        int hl = Math.max(1, p.amaxHistory);
        history = new double[LAYERS][3][hl];
        historyCount = new int[LAYERS][3];
    }

    @Override
    public NumericPolicy policy() {
        return p;
    }

    @Override
    public long[] clampCounts() {
        return new long[]{clampedX, castX, clampedW, castW, clampedG, castG};
    }

    /**
     * Mean cross-entropy over the batch, with gradients written into {@code g}.
     *
     * @param contexts {@code batch * ctx} token ids
     * @param targets  {@code batch} token ids
     */
    @Override
    public double lossAndGrads(int[] contexts, int[] targets, Grads g) {
        double loss = forward(contexts, targets);
        backward(contexts, g);
        return loss;
    }

    /** Mean cross-entropy only. Also leaves the softmax gradient ready for backward. */
    @Override
    public double forward(int[] contexts, int[] targets) {
        Net n = net;
        int e = n.emb;
        int h = n.hidden;

        for (int i = 0; i < batch; i++) {
            for (int c = 0; c < n.ctx; c++) {
                System.arraycopy(n.embed, contexts[i * n.ctx + c] * e, x0, i * n.in1 + c * e, e);
            }
        }

        linearForward(0, x0, n.in1, n.w1, n.b1, h, h1);
        tanh(h1);
        linearForward(1, h1, h, n.w2, n.b2, h, h2);
        tanh(h2);
        linearForward(2, h2, h, n.w3, n.b3, n.vocab, logits);

        double loss = 0.0;
        int v = n.vocab;
        for (int i = 0; i < batch; i++) {
            int off = i * v;
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < v; k++) {
                max = Math.max(max, logits[off + k]);
            }
            double z = 0.0;
            for (int k = 0; k < v; k++) {
                z += Math.exp(logits[off + k] - max);
            }
            double logZ = Math.log(z) + max;
            loss += logZ - logits[off + targets[i]];
            for (int k = 0; k < v; k++) {
                dLogits[off + k] = Math.exp(logits[off + k] - logZ) / batch;
            }
            dLogits[off + targets[i]] -= 1.0 / batch;
        }
        return loss / batch;
    }

    private void backward(int[] contexts, Grads g) {
        Net n = net;
        int h = n.hidden;
        int e = n.emb;

        g.zero();

        linearBackward(2, dLogits, h, n.vocab, dh2, g.get(5), g.get(6));
        tanhBackward(h2, dh2);
        linearBackward(1, dh2, h, h, dh1, g.get(3), g.get(4));
        tanhBackward(h1, dh1);
        linearBackward(0, dh1, n.in1, h, dx0, g.get(1), g.get(2));

        double[] embedGrad = g.get(0);
        for (int i = 0; i < batch; i++) {
            for (int c = 0; c < n.ctx; c++) {
                int tok = contexts[i * n.ctx + c];
                int src = i * n.in1 + c * e;
                int dst = tok * e;
                for (int k = 0; k < e; k++) {
                    embedGrad[dst + k] += dx0[src + k];
                }
            }
        }
    }

    // --------------------------------------------------------------- linear

    private void linearForward(int l, double[] x, int in, double[] w, double[] b, int out, double[] y) {
        MiniFloat f = p.inputFormat;
        int nx = batch * in;
        int nw = out * in;

        double sx = scaleFor(l, T_X, f, x, nx);
        double sw = scaleFor(l, T_W, f, w, nw);
        clampedX += Quant.quantize(x, 0, qX[l], 0, nx, f, p, sx, rng);
        clampedW += Quant.quantize(w, 0, qW[l], 0, nw, f, p, sw, rng);
        if (f != null) {
            castX += nx;
            castW += nw;
        }

        double[] qx = qX[l];
        double[] qw = qW[l];
        for (int i = 0; i < batch; i++) {
            for (int o = 0; o < out; o++) {
                y[i * out + o] = Dot.dot(qx, i * in, 1, qw, o * in, 1, in, p, rng) + b[o];
            }
        }
    }

    private void linearBackward(int l, double[] dy, int in, int out, double[] dx, double[] dw, double[] db) {
        MiniFloat f = p.gradCastFormat();
        int ng = batch * out;

        double sg = scaleFor(l, T_G, f, dy, ng);
        clampedG += Quant.quantize(dy, 0, qG[l], 0, ng, f, p, p.gradCastRounding(), sg, rng);
        if (f != null) {
            castG += ng;
        }

        double[] qg = qG[l];
        double[] qx = qX[l];
        double[] qw = qW[l];

        // dX = dY W: row i of dY against column k of W.
        for (int i = 0; i < batch; i++) {
            for (int k = 0; k < in; k++) {
                dx[i * in + k] = Dot.dot(qg, i * out, 1, qw, k, in, out, p, rng);
            }
        }
        // dW = dY^T X: column o of dY against column k of X, summed over the batch.
        for (int o = 0; o < out; o++) {
            for (int k = 0; k < in; k++) {
                dw[o * in + k] = Dot.dot(qg, o, out, qx, k, in, batch, p, rng);
            }
        }
        // Bias gradients are sums of the uncast gradient, kept in master precision.
        for (int o = 0; o < out; o++) {
            double s = 0.0;
            for (int i = 0; i < batch; i++) {
                s += dy[i * out + o];
            }
            db[o] = s;
        }
    }

    /**
     * The per-tensor factor for this cast. Current scaling uses this tensor's amax.
     * Delayed scaling uses the largest amax over the previous {@code amaxHistory}
     * steps, then records this one; a tensor that grows faster than its history
     * therefore saturates, which is the failure delayed scaling is known for.
     */
    private double scaleFor(int l, int t, MiniFloat f, double[] x, int n) {
        if (f == null || p.scaling != Scaling.PER_TENSOR) {
            return 1.0;
        }
        double now = Quant.amax(x, 0, n);
        if (p.amaxHistory <= 0) {
            return Quant.perTensorScale(f, now);
        }
        double[] hist = history[l][t];
        int count = historyCount[l][t];
        double use;
        if (count == 0) {
            use = now;
        } else {
            use = 0.0;
            for (int i = 0; i < Math.min(count, hist.length); i++) {
                use = Math.max(use, hist[i]);
            }
        }
        hist[count % hist.length] = now;
        historyCount[l][t] = count + 1;
        return Quant.perTensorScale(f, use);
    }

    private static void tanh(double[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.tanh(a[i]);
        }
    }

    private static void tanhBackward(double[] out, double[] grad) {
        for (int i = 0; i < grad.length; i++) {
            grad[i] *= 1.0 - out[i] * out[i];
        }
    }
}
