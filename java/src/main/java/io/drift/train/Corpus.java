package io.drift.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.TreeSet;

/**
 * Character-level text, split into a training span and a held-out span.
 *
 * <p>The vocabulary is every distinct character, sorted, so it is a pure function of
 * the file and two runs on the same file index characters identically.
 */
public final class Corpus {

    public final int[] train;
    public final int[] val;
    public final int vocab;

    private Corpus(int[] train, int[] val, int vocab) {
        this.train = train;
        this.val = val;
        this.vocab = vocab;
    }

    public static Corpus load(Path path, double valFraction) throws IOException {
        String text = Files.readString(path, StandardCharsets.UTF_8);
        TreeSet<Character> chars = new TreeSet<>();
        for (int i = 0; i < text.length(); i++) {
            chars.add(text.charAt(i));
        }
        int[] index = new int[Character.MAX_VALUE + 1];
        int next = 0;
        for (char c : chars) {
            index[c] = next++;
        }
        int[] all = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            all[i] = index[text.charAt(i)];
        }
        // The held-out span is the tail of the file rather than a random sample, so no
        // training window can overlap it.
        int split = (int) (all.length * (1.0 - valFraction));
        int[] train = java.util.Arrays.copyOfRange(all, 0, split);
        int[] val = java.util.Arrays.copyOfRange(all, split, all.length);
        return new Corpus(train, val, next);
    }
}
