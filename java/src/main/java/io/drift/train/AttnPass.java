package io.drift.train;

import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;
import io.drift.kernel.Dot;
import io.drift.kernel.NumericPolicy;
import io.drift.kernel.NumericPolicy.Scaling;
import io.drift.kernel.Quant;

/**
 * Forward and backward for {@link AttnNet} under a numeric policy.
 *
 * <p>Every matmul goes through the policy, including the two an MLP does not have.
 * The score matmul casts a query and a key, both activations. The context matmul
 * casts the softmax probabilities and the values, again both activations. So in this
 * model a cast error can enter on both sides of a product, which is the thing the
 * MLP could not show.
 *
 * <p>Scaling here is always computed from the tensor in hand. Delayed scaling, which
 * the MLP supports through an amax history, is not wired up for this model yet, so a
 * policy that asks for it gets current scaling and the run records that in its name
 * only. Everything outside the matmuls stays in double: the embedding lookup, the
 * residual adds, the softmax, the ReLU and the loss.
 */
public final class AttnPass implements ModelPass {

    public final NumericPolicy p;
    private final AttnNet net;
    private final Rng rng;
    private final int batch;
    private final int t;
    private final int d;
    private final int h;
    private final int v;
    private final int rows;
    private final double scoreScale;

    private final double[] x;
    private final double[] q;
    private final double[] k;
    private final double[] val;
    private final double[] scores;
    private final double[] prob;
    private final double[] ctxv;
    private final double[] attnOut;
    private final double[] hres;
    private final double[] z1;
    private final double[] m1;
    private final double[] y;
    private final double[] logits;

    private final double[] dLogits;
    private final double[] dy;
    private final double[] dh;
    private final double[] dm1;
    private final double[] dz1;
    private final double[] dAttnOut;
    private final double[] dCtxv;
    private final double[] dProb;
    private final double[] dScore;
    private final double[] dq;
    private final double[] dk;
    private final double[] dv;
    private final double[] dx;

    // Cast copies, kept from the forward pass so the backward pass reuses them.
    private final double[] qx;
    private final double[] qq;
    private final double[] qk;
    private final double[] qv;
    private final double[] qprob;
    private final double[] qctxv;
    private final double[] qh;
    private final double[] qm1;
    private final double[] qy;
    private final double[] qgRows;
    private final double[] qgScore;
    private final double[] qWq;
    private final double[] qWk;
    private final double[] qWv;
    private final double[] qWo;
    private final double[] qW1;
    private final double[] qW2;
    private final double[] qWout;
    private final double[] qgWide;

    public long clampedX;
    public long clampedW;
    public long clampedG;
    public long castX;
    public long castW;
    public long castG;

    public AttnPass(NumericPolicy p, AttnNet net, int batch, long rngSeed) {
        this.p = p;
        this.net = net;
        this.batch = batch;
        this.rng = new Rng(rngSeed);
        this.t = net.ctx;
        this.d = net.dim;
        this.h = net.hidden;
        this.v = net.vocab;
        this.rows = batch * t;
        this.scoreScale = 1.0 / Math.sqrt(d);

        x = new double[rows * d];
        q = new double[rows * d];
        k = new double[rows * d];
        val = new double[rows * d];
        scores = new double[batch * t * t];
        prob = new double[batch * t * t];
        ctxv = new double[rows * d];
        attnOut = new double[rows * d];
        hres = new double[rows * d];
        z1 = new double[rows * h];
        m1 = new double[rows * h];
        y = new double[rows * d];
        logits = new double[rows * v];

        dLogits = new double[rows * v];
        dy = new double[rows * d];
        dh = new double[rows * d];
        dm1 = new double[rows * h];
        dz1 = new double[rows * h];
        dAttnOut = new double[rows * d];
        dCtxv = new double[rows * d];
        dProb = new double[batch * t * t];
        dScore = new double[batch * t * t];
        dq = new double[rows * d];
        dk = new double[rows * d];
        dv = new double[rows * d];
        dx = new double[rows * d];

        qx = new double[rows * d];
        qq = new double[rows * d];
        qk = new double[rows * d];
        qv = new double[rows * d];
        qprob = new double[batch * t * t];
        qctxv = new double[rows * d];
        qh = new double[rows * d];
        qm1 = new double[rows * h];
        qy = new double[rows * d];
        qgRows = new double[rows * Math.max(Math.max(d, h), v)];
        qgScore = new double[batch * t * t];
        qgWide = new double[rows * Math.max(Math.max(d, h), v)];
        qWq = new double[d * d];
        qWk = new double[d * d];
        qWv = new double[d * d];
        qWo = new double[d * d];
        qW1 = new double[h * d];
        qW2 = new double[d * h];
        qWout = new double[v * d];
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
     * The attention probabilities from the last forward pass, laid out as batch by
     * query position by key position. Exposed so a test can check the causal mask
     * and that each row sums to one.
     */
    public double[] attentionProbabilities() {
        return prob;
    }

    @Override
    public double lossAndGrads(int[] contexts, int[] targets, Grads g) {
        double loss = forward(contexts, targets);
        backward(contexts, g);
        return loss;
    }

    @Override
    public double forward(int[] contexts, int[] targets) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                int tok = contexts[b * t + i];
                int dst = (b * t + i) * d;
                for (int c = 0; c < d; c++) {
                    x[dst + c] = net.embed[tok * d + c] + net.pos[i * d + c];
                }
            }
        }

        castAct(x, qx, rows * d);
        castWeight(net.wq, qWq, d * d);
        castWeight(net.wk, qWk, d * d);
        castWeight(net.wv, qWv, d * d);
        linear(qx, qWq, null, rows, d, d, q);
        linear(qx, qWk, null, rows, d, d, k);
        linear(qx, qWv, null, rows, d, d, val);

        // Scores: a product of two activations, which an MLP never has.
        castAct(q, qq, rows * d);
        castAct(k, qk, rows * d);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                int base = b * t * t + i * t;
                for (int s = 0; s <= i; s++) {
                    scores[base + s] = Dot.dot(qq, (b * t + i) * d, 1, qk, (b * t + s) * d, 1, d, p, rng)
                            * scoreScale;
                }
                for (int s = i + 1; s < t; s++) {
                    scores[base + s] = 0.0;
                    prob[base + s] = 0.0;
                }
                double max = Double.NEGATIVE_INFINITY;
                for (int s = 0; s <= i; s++) {
                    max = Math.max(max, scores[base + s]);
                }
                double z = 0.0;
                for (int s = 0; s <= i; s++) {
                    double e = Math.exp(scores[base + s] - max);
                    prob[base + s] = e;
                    z += e;
                }
                for (int s = 0; s <= i; s++) {
                    prob[base + s] /= z;
                }
            }
        }

        // Context: probabilities against values, again two activations.
        castAct(prob, qprob, batch * t * t);
        castAct(val, qv, rows * d);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                for (int c = 0; c < d; c++) {
                    ctxv[(b * t + i) * d + c] =
                            Dot.dot(qprob, b * t * t + i * t, 1, qv, b * t * d + c, d, i + 1, p, rng);
                }
            }
        }

        castAct(ctxv, qctxv, rows * d);
        castWeight(net.wo, qWo, d * d);
        linear(qctxv, qWo, null, rows, d, d, attnOut);

        for (int i = 0; i < rows * d; i++) {
            hres[i] = x[i] + attnOut[i];
        }

        castAct(hres, qh, rows * d);
        castWeight(net.w1, qW1, h * d);
        linear(qh, qW1, net.b1, rows, d, h, z1);
        for (int i = 0; i < rows * h; i++) {
            m1[i] = Math.max(0.0, z1[i]);
        }
        castAct(m1, qm1, rows * h);
        castWeight(net.w2, qW2, d * h);
        linear(qm1, qW2, net.b2, rows, h, d, y);
        for (int i = 0; i < rows * d; i++) {
            y[i] += hres[i];
        }

        castAct(y, qy, rows * d);
        castWeight(net.wout, qWout, v * d);
        linear(qy, qWout, net.bout, rows, d, v, logits);

        double loss = 0.0;
        for (int r = 0; r < rows; r++) {
            int off = r * v;
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < v; c++) {
                max = Math.max(max, logits[off + c]);
            }
            double z = 0.0;
            for (int c = 0; c < v; c++) {
                z += Math.exp(logits[off + c] - max);
            }
            double logZ = Math.log(z) + max;
            loss += logZ - logits[off + targets[r]];
            for (int c = 0; c < v; c++) {
                dLogits[off + c] = Math.exp(logits[off + c] - logZ) / rows;
            }
            dLogits[off + targets[r]] -= 1.0 / rows;
        }
        return loss / rows;
    }

    private void backward(int[] contexts, Grads g) {
        g.zero();

        // Output head.
        castGrad(dLogits, qgWide, rows * v);
        dInput(qgWide, qWout, rows, d, v, dy, false);
        dWeight(qgWide, qy, rows, d, v, g.get(10));
        dBias(dLogits, rows, v, g.get(11));

        // MLP with its residual: y = h + mlp(h), so dh gets both paths.
        System.arraycopy(dy, 0, dh, 0, rows * d);
        castGrad(dy, qgRows, rows * d);
        dInput(qgRows, qW2, rows, h, d, dm1, false);
        dWeight(qgRows, qm1, rows, h, d, g.get(8));
        dBias(dy, rows, d, g.get(9));

        for (int i = 0; i < rows * h; i++) {
            dz1[i] = z1[i] > 0.0 ? dm1[i] : 0.0;
        }
        castGrad(dz1, qgWide, rows * h);
        dInput(qgWide, qW1, rows, d, h, dh, true);
        dWeight(qgWide, qh, rows, d, h, g.get(6));
        dBias(dz1, rows, h, g.get(7));

        // Attention output projection, and the residual around the block.
        System.arraycopy(dh, 0, dAttnOut, 0, rows * d);
        System.arraycopy(dh, 0, dx, 0, rows * d);
        castGrad(dAttnOut, qgRows, rows * d);
        dInput(qgRows, qWo, rows, d, d, dCtxv, false);
        dWeight(qgRows, qctxv, rows, d, d, g.get(5));

        // Context matmul: probabilities and values.
        castGrad(dCtxv, qgWide, rows * d);
        java.util.Arrays.fill(dv, 0.0);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                int base = b * t * t + i * t;
                for (int s = 0; s <= i; s++) {
                    dProb[base + s] = Dot.dot(qgWide, (b * t + i) * d, 1, qv, (b * t + s) * d, 1, d, p, rng);
                }
                for (int s = i + 1; s < t; s++) {
                    dProb[base + s] = 0.0;
                }
            }
            for (int s = 0; s < t; s++) {
                for (int c = 0; c < d; c++) {
                    dv[(b * t + s) * d + c] = Dot.dot(qprob, b * t * t + s * t + s, t,
                            qgWide, (b * t + s) * d + c, d, t - s, p, rng);
                }
            }
        }

        // Softmax, in double, then the score scale.
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                int base = b * t * t + i * t;
                double sum = 0.0;
                for (int s = 0; s <= i; s++) {
                    sum += prob[base + s] * dProb[base + s];
                }
                for (int s = 0; s <= i; s++) {
                    dScore[base + s] = prob[base + s] * (dProb[base + s] - sum) * scoreScale;
                }
                for (int s = i + 1; s < t; s++) {
                    dScore[base + s] = 0.0;
                }
            }
        }

        // Score matmul: queries against keys.
        castGrad(dScore, qgScore, batch * t * t);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                for (int c = 0; c < d; c++) {
                    dq[(b * t + i) * d + c] = Dot.dot(qgScore, b * t * t + i * t, 1,
                            qk, b * t * d + c, d, i + 1, p, rng);
                }
            }
            for (int s = 0; s < t; s++) {
                for (int c = 0; c < d; c++) {
                    dk[(b * t + s) * d + c] = Dot.dot(qgScore, b * t * t + s * t + s, t,
                            qq, (b * t + s) * d + c, d, t - s, p, rng);
                }
            }
        }

        // Query, key and value projections. Each adds into dx.
        projBackward(dq, qWq, g.get(2));
        projBackward(dk, qWk, g.get(3));
        projBackward(dv, qWv, g.get(4));

        double[] embedGrad = g.get(0);
        double[] posGrad = g.get(1);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < t; i++) {
                int tok = contexts[b * t + i];
                int src = (b * t + i) * d;
                for (int c = 0; c < d; c++) {
                    embedGrad[tok * d + c] += dx[src + c];
                    posGrad[i * d + c] += dx[src + c];
                }
            }
        }
    }

    /** One projection backward: weight gradient, and the input gradient added into dx. */
    private void projBackward(double[] dOut, double[] qw, double[] dw) {
        castGrad(dOut, qgRows, rows * d);
        dInput(qgRows, qw, rows, d, d, dx, true);
        dWeight(qgRows, qx, rows, d, d, dw);
    }

    // ------------------------------------------------------------- matmuls

    private void linear(double[] qin, double[] qw, double[] bias, int r, int in, int out, double[] outBuf) {
        for (int i = 0; i < r; i++) {
            for (int o = 0; o < out; o++) {
                outBuf[i * out + o] = Dot.dot(qin, i * in, 1, qw, o * in, 1, in, p, rng)
                        + (bias == null ? 0.0 : bias[o]);
            }
        }
    }

    /** dX = dY W, optionally added to what is already in dxBuf. */
    private void dInput(double[] qg, double[] qw, int r, int in, int out, double[] dxBuf, boolean accumulate) {
        for (int i = 0; i < r; i++) {
            for (int c = 0; c < in; c++) {
                double s = Dot.dot(qg, i * out, 1, qw, c, in, out, p, rng);
                if (accumulate) {
                    dxBuf[i * in + c] += s;
                } else {
                    dxBuf[i * in + c] = s;
                }
            }
        }
    }

    /** dW = dY transpose times X, summed over rows. */
    private void dWeight(double[] qg, double[] qin, int r, int in, int out, double[] dw) {
        for (int o = 0; o < out; o++) {
            for (int c = 0; c < in; c++) {
                dw[o * in + c] = Dot.dot(qg, o, out, qin, c, in, r, p, rng);
            }
        }
    }

    /** Bias gradients are sums of the uncast gradient, kept in master precision. */
    private void dBias(double[] dOut, int r, int out, double[] db) {
        for (int o = 0; o < out; o++) {
            double s = 0.0;
            for (int i = 0; i < r; i++) {
                s += dOut[i * out + o];
            }
            db[o] = s;
        }
    }

    // --------------------------------------------------------------- casts

    private void castAct(double[] src, double[] dst, int n) {
        MiniFloat f = p.inputFormat;
        clampedX += cast(src, dst, n, f, p.inputRounding);
        if (f != null) {
            castX += n;
        }
    }

    private void castWeight(double[] src, double[] dst, int n) {
        MiniFloat f = p.inputFormat;
        clampedW += cast(src, dst, n, f, p.inputRounding);
        if (f != null) {
            castW += n;
        }
    }

    private void castGrad(double[] src, double[] dst, int n) {
        MiniFloat f = p.gradCastFormat();
        clampedG += cast(src, dst, n, f, p.gradCastRounding());
        if (f != null) {
            castG += n;
        }
    }

    private int cast(double[] src, double[] dst, int n, MiniFloat f, Rounding rm) {
        double scale = 1.0;
        if (f != null && p.scaling == Scaling.PER_TENSOR) {
            scale = Quant.perTensorScale(f, Quant.amax(src, 0, n));
        }
        return Quant.quantize(src, 0, dst, 0, n, f, p, rm, scale, rng);
    }
}
