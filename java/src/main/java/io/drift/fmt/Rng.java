package io.drift.fmt;

/**
 * splitmix64. A mutable long of state, no allocation, and fast enough to sit inside
 * a stochastic-rounding inner loop where it is called once per multiply-accumulate.
 *
 * <p>Deliberately not {@code java.util.Random}: that synchronises on its seed, which
 * turns every rounding decision into a contended atomic.
 */
public final class Rng {
    private long state;

    public Rng(long seed) {
        this.state = seed;
    }

    public long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Uniform in [0, 1). 53 bits, so it can resolve any fraction these formats produce. */
    public double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    public long state() {
        return state;
    }

    public void reseed(long seed) {
        this.state = seed;
    }
}
