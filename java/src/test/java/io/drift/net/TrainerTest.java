package io.drift.net;

import io.drift.kernel.NumericPolicy;
import io.drift.train.Corpus;
import io.drift.train.Policies;
import io.drift.train.Trainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainerTest {

    private static Trainer.Config tiny() {
        Trainer.Config cfg = new Trainer.Config();
        cfg.ctx = 4;
        cfg.emb = 4;
        cfg.hidden = 8;
        cfg.batch = 4;
        cfg.steps = 60;
        cfg.measureEvery = 10;
        cfg.evalEvery = 30;
        cfg.evalBatches = 2;
        return cfg;
    }

    private static Corpus corpus(Path dir) throws IOException {
        Path text = dir.resolve("corpus.txt");
        Files.writeString(text, "the quick brown fox jumps over the lazy dog. ".repeat(200));
        return Corpus.load(text, 0.1);
    }

    /** A probe observes the run. If it could change it, every probe result would be suspect. */
    @Test
    void probesDoNotChangeTheRun(@TempDir Path dir) throws IOException {
        Corpus c = corpus(dir);
        NumericPolicy p = Policies.byName("mxfp4-sr-bwd");

        Trainer.Config plain = tiny();
        Trainer.Summary a = new Trainer(plain, c).run(p, 1, dir.resolve("a.jsonl"));

        Trainer.Config probed = tiny();
        probed.probes = new String[]{"mxfp4", "mxfp4-sr-bwd"};
        Trainer.Summary b = new Trainer(probed, c).run(p, 1, dir.resolve("b.jsonl"));

        assertEquals(a.finalTrainLoss(), b.finalTrainLoss());
        assertEquals(a.finalValLoss(), b.finalValLoss());
        assertEquals(a.meanGain(), b.meanGain());
    }

    /**
     * mxfp4 uses no random numbers, so probing a run with its own policy must reproduce
     * that run's gradient measurement digit for digit. This checks the probe is computed
     * at the run's weights and batch rather than somewhere else.
     */
    @Test
    void deterministicSelfProbeMatchesTheRun(@TempDir Path dir) throws IOException {
        Corpus c = corpus(dir);
        Trainer.Config cfg = tiny();
        cfg.probes = new String[]{"mxfp4"};
        Path out = dir.resolve("self.jsonl");
        new Trainer(cfg, c).run(Policies.byName("mxfp4"), 2, out);

        Pattern top = Pattern.compile("\"kind\":\"measure\".*?\"gain\":([^,]+),");
        Pattern probe = Pattern.compile("\"probes\":\\{\"mxfp4\":\\{\"cos\":[^,]+,\"gain\":([^,]+),");
        List<String> lines = Files.readAllLines(out);
        int checked = 0;
        for (String line : lines) {
            if (!line.contains("\"kind\":\"measure\"")) {
                continue;
            }
            Matcher t = top.matcher(line);
            Matcher pr = probe.matcher(line);
            assertTrue(t.find() && pr.find(), line);
            assertEquals(t.group(1), pr.group(1), line);
            checked++;
        }
        assertTrue(checked >= 6, "measurements checked: " + checked);
    }
}
