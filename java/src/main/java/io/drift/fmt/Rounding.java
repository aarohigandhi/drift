package io.drift.fmt;

/**
 * How a real value is mapped onto the two representable neighbours that bracket it.
 *
 * <p>The choice matters more than it looks. {@link #NEAREST_EVEN} is the smallest
 * error per operation, but the error is a deterministic function of the value, so a
 * quantity that is updated repeatedly by a sub-quantum amount never moves at all:
 * every update rounds back to where it started. {@link #STOCHASTIC} has larger error
 * per operation and zero bias in expectation, so the same update does move, in
 * proportion to its size. Which of those two properties you want depends on whether
 * the number is a product (want small error) or an accumulator (want no bias).
 */
public enum Rounding {
    /** Round half to even. IEEE 754's default and every GPU's default. */
    NEAREST_EVEN,
    /** Round up with probability equal to the fractional distance. Unbiased. */
    STOCHASTIC,
    /** Round toward zero. Included because it is the cheapest in hardware. */
    TRUNCATE
}
