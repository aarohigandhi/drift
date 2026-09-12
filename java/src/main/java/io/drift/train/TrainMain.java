package io.drift.train;

import io.drift.kernel.NumericPolicy;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Runs the sweep: every requested policy at every requested seed, one run per
 * thread. Runs share nothing but the corpus, which is read-only.
 *
 * <pre>
 * --data ../data/corpus.txt  --out ../results/runs
 * --policies all | fp8,mxfp4,...  --seeds 1,2,3  --steps 3000  --threads 9
 * </pre>
 */
public final class TrainMain {

    public static void main(String[] args) throws Exception {
        String data = "../data/corpus.txt";
        String outDir = "../results/runs";
        String policies = "all";
        String seeds = "1";
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        Trainer.Config cfg = new Trainer.Config();

        for (int i = 0; i < args.length; i += 2) {
            String v = args[i + 1];
            switch (args[i]) {
                case "--data" -> data = v;
                case "--out" -> outDir = v;
                case "--policies" -> policies = v;
                case "--seeds" -> seeds = v;
                case "--steps" -> cfg.steps = Integer.parseInt(v);
                case "--threads" -> threads = Integer.parseInt(v);
                case "--hidden" -> cfg.hidden = Integer.parseInt(v);
                case "--batch" -> cfg.batch = Integer.parseInt(v);
                default -> throw new IllegalArgumentException("unknown flag " + args[i]);
            }
        }

        Corpus corpus = Corpus.load(Path.of(data), 0.1);
        System.out.printf(Locale.ROOT, "corpus: %,d train chars, %,d held out, vocab %d%n",
                corpus.train.length, corpus.val.length, corpus.vocab);

        List<NumericPolicy> ps = new ArrayList<>();
        if (policies.equals("all")) {
            ps.addAll(Policies.all().values());
        } else {
            for (String name : policies.split(",")) {
                ps.add(Policies.byName(name.trim()));
            }
        }
        long[] seedList = Arrays.stream(seeds.split(",")).mapToLong(Long::parseLong).toArray();

        Trainer trainer = new Trainer(cfg, corpus);
        Path out = Path.of(outDir);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Trainer.Summary>> futures = new ArrayList<>();
        for (long seed : seedList) {
            for (NumericPolicy p : ps) {
                futures.add(pool.submit(() -> {
                    Trainer.Summary s = trainer.run(p, seed, out.resolve(p.name + "-s" + seed + ".jsonl"));
                    System.out.printf(Locale.ROOT,
                            "%-24s seed %d  %s  train %.4f  val %.4f  cos %.6f  gain %.4f  rel %.3e  %.0fs%n",
                            s.policy(), s.seed(), s.diverged() ? "DIVERGED" : "ok      ",
                            s.finalTrainLoss(), s.finalValLoss(), s.meanCos(), s.meanGain(),
                            s.meanRel(), s.seconds());
                    return s;
                }));
            }
        }

        Files.createDirectories(out);
        try (BufferedWriter w = Files.newBufferedWriter(out.resolve("summary.csv"))) {
            w.write("policy,seed,steps,diverged,train_loss,val_loss,mean_cos,mean_gain,mean_rel,seconds\n");
            for (Future<Trainer.Summary> f : futures) {
                Trainer.Summary s = f.get();
                w.write(String.format(Locale.ROOT, "%s,%d,%d,%b,%.6f,%.6f,%.8f,%.6f,%.6e,%.1f%n",
                        s.policy(), s.seed(), s.steps(), s.diverged(), s.finalTrainLoss(),
                        s.finalValLoss(), s.meanCos(), s.meanGain(), s.meanRel(), s.seconds()));
            }
        }
        pool.shutdown();
    }

    private TrainMain() {
    }
}
