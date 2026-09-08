package io.drift.tools;

import io.drift.fmt.Formats;
import io.drift.fmt.MiniFloat;
import io.drift.fmt.Rng;
import io.drift.fmt.Rounding;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Writes out what the Java formats believe, so that something with no shared code can
 * check it.
 *
 * <p>For every narrow format this dumps three tables: the value of each of its bit
 * patterns, the code produced by rounding the product of each ordered pair of codes,
 * and the same for the sum. An eight bit format has 256 codes and so 65,536 pairs,
 * which is small enough to enumerate completely. {@code python/verify_oracle.py}
 * recomputes all of it in exact rational arithmetic from an independent reading of
 * the specification, and any disagreement is a bug in one of the two.
 *
 * <p>FP16 and BF16 have 65,536 codes, so their pair tables would have 4.3 billion
 * entries. Their values are still dumped exhaustively; their arithmetic is sampled.
 */
public final class DumpTables {

    /** Above this width, enumerate values but sample arithmetic. */
    private static final int EXHAUSTIVE_PAIR_WIDTH = 8;

    /**
     * Kept modest on purpose. The checking side does this arithmetic in exact
     * rationals, which is thousands of times slower than the doubles that produced
     * it, and a wider sample would buy very little: the narrow formats are already
     * covered completely, and these two are the ones whose rounding path is shared
     * with them.
     */
    private static final int SAMPLED_PAIRS = 200_000;

    public static void main(String[] args) throws IOException {
        Path out = Path.of(args.length > 0 ? args[0] : "../results/oracle");
        Files.createDirectories(out);

        for (MiniFloat f : Formats.ALL) {
            dumpValues(f, out.resolve("values_" + f.name + ".csv"));
            if (f.width <= EXHAUSTIVE_PAIR_WIDTH) {
                dumpPairsExhaustive(f, out.resolve("pairs_" + f.name + ".csv"));
            } else {
                dumpPairsSampled(f, out.resolve("pairs_" + f.name + ".csv"));
            }
        }
        System.out.println("wrote oracle tables to " + out.toAbsolutePath().normalize());
    }

    /**
     * One row per bit pattern. The value travels as the raw bits of the double rather
     * than as decimal text, so nothing is lost on the way across.
     */
    private static void dumpValues(MiniFloat f, Path path) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write("# format=" + f.name
                    + " expBits=" + f.expBits
                    + " mantBits=" + f.mantBits
                    + " bias=" + f.bias
                    + " specials=" + f.specials
                    + " overflow=" + f.overflow
                    + " codes=" + f.codeCount() + "\n");
            w.write("code,double_bits_hex,value\n");
            for (int c = 0; c < f.codeCount(); c++) {
                double v = f.decode(c);
                w.write(c + ",");
                w.write(Long.toHexString(Double.doubleToRawLongBits(v)));
                w.write(",");
                w.write(describe(v));
                w.write("\n");
            }
        }
    }

    /**
     * Every ordered pair, with the code that rounding their exact product and their
     * exact sum lands on. Both operands are already representable, so the only
     * approximation in the row is the single rounding under test.
     */
    private static void dumpPairsExhaustive(MiniFloat f, Path path) throws IOException {
        int n = f.codeCount();
        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write("# format=" + f.name + " mode=exhaustive pairs=" + ((long) n * n) + "\n");
            w.write("a,b,prod_code,sum_code\n");
            for (int a = 0; a < n; a++) {
                double av = f.decode(a);
                for (int b = 0; b < n; b++) {
                    double bv = f.decode(b);
                    w.write(a + "," + b + "," + code(f, av * bv) + "," + code(f, av + bv) + "\n");
                }
            }
        }
    }

    /**
     * The same table for formats too wide to enumerate. The pairs are drawn from a
     * fixed seed so the row set is reproducible.
     */
    private static void dumpPairsSampled(MiniFloat f, Path path) throws IOException {
        int n = f.codeCount();
        Rng rng = new Rng(0x51ED_1EA5L);
        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write("# format=" + f.name + " mode=sampled seed=0x51ED1EA5 pairs=" + SAMPLED_PAIRS + "\n");
            w.write("a,b,prod_code,sum_code\n");
            for (int i = 0; i < SAMPLED_PAIRS; i++) {
                int a = (int) ((rng.nextLong() >>> 1) % n);
                int b = (int) ((rng.nextLong() >>> 1) % n);
                double av = f.decode(a);
                double bv = f.decode(b);
                w.write(a + "," + b + "," + code(f, av * bv) + "," + code(f, av + bv) + "\n");
            }
        }
    }

    /**
     * The exact product or sum of two representable values can need more mantissa bits
     * than a double has only if the format is wider than 26 bits, which none of these
     * are, so the double arithmetic here is exact and the only rounding is the encode.
     */
    private static int code(MiniFloat f, double exact) {
        return f.encode(exact, Rounding.NEAREST_EVEN, null);
    }

    private static String describe(double v) {
        if (Double.isNaN(v)) {
            return "nan";
        }
        if (Double.isInfinite(v)) {
            return v > 0 ? "inf" : "-inf";
        }
        return String.format(Locale.ROOT, "%.17g", v);
    }
}
