# Where this is

## The question

When a low precision training run goes wrong, the cause is usually named from a
short list — not enough exponent bits, a stale scaling factor, the wrong rounding
mode — and almost never measured. On real hardware it cannot be: you cannot hold
the format fixed and change only the order the products were added in, because the
order is burned into the kernel.

In software you can. That is the whole reason this exists. Every arithmetic
decision becomes an independent variable, and a run that diverges can be attributed
to one of them instead of blamed on the format.

## Built and verified

The formats and the oracle.

`io.drift.fmt.MiniFloat` is one parameterised implementation of every narrow float
that matters: E4M3, E5M2, FP4 E2M1, both FP6 variants, FP16, BF16. Rounding is
exact rather than approximate — the quantum at any exponent is a power of two, so
rescaling a double by it loses nothing, and the only approximation is the single
deliberate rounding.

`python/drift/minifloat.py` reaches the same answers a different way, in
`fractions.Fraction`, with no floating point anywhere in its path and no code
shared with the Java side. The field widths come from IEEE 754 and the OCP
Microscaling spec, read independently.

`./gradlew dumpTables` then `python verify_oracle.py` compares them:

| format | codes | pairs checked | result |
|---|---:|---:|---|
| e4m3 | 256 | 65,536 | ok |
| e5m2 | 256 | 65,536 | ok |
| e2m1 | 16 | 256 | ok |
| e3m2 | 64 | 4,096 | ok |
| e2m3 | 64 | 4,096 | ok |
| fp16 | 65,536 | 200,000 | ok |
| bf16 | 65,536 | 200,000 | ok |

For the 8-bit and 6-bit formats that is every ordered pair, so the agreement covers
the entire arithmetic of the format rather than a sample of it. Each pair is
checked twice, once for the product and once for the sum.

The check earned its keep immediately. It found 154 disagreements in e4m3 and the
same pattern in e5m2, all of them negative zero: a positive zero times a negative
number is negative zero, every one of these formats spends a bit pattern on it, and
a `Fraction` has no way to hold it. The Java side was right and the reference was
wrong. Fixed by giving the reference an explicit negative zero.

## Next

1. **The kernel.** A dot product whose accumulator is an independent variable:
   width (fp64 / fp32 / fp16 / bf16), order (sequential, reversed, pairwise,
   blocked in chunks of k, which is what a tensor core actually does), and rounding
   mode. Products stay exact, as they do in hardware — only the accumulation
   rounds. Plus the scaling policies: none, per-tensor current, per-tensor delayed
   with an amax history, and MX per-block with an E8M0 scale.

   One thing already settled: MX block scales are exact powers of two, so
   `scale * element` is exact and a block-scaled tensor can be stored dequantized
   without changing a single product. Per-tensor scales are arbitrary floats and
   cannot.

2. **A small transformer in Java**, trained on real text, with every matmul routed
   through the policy above.

3. **The instrument.** At intervals, recompute the gradient from the same weights
   in fp64 and compare: cosine similarity against the exact gradient, and the mean
   of the error. Those two separate noise from bias, which is the distinction the
   folklore keeps eliding — stochastic rounding is unbiased per operation, but the
   accumulator it feeds is not rounded stochastically.

4. **The sweep.** Baseline plus one-factor-at-a-time, plus the cells where two
   factors are expected to interact. Then report what it says.

## Not yet done

Nothing is claimed about training yet — none has been run.

There are no JUnit tests yet either. The oracle check is a stronger statement than
any unit test would be, but it runs across a language boundary and needs the tables
built first, so the format code should still get a test source set that CI can run
on its own.
