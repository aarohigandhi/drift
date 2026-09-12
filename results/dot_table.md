| group | policy | n=256 | n=1024 | n=4096 | clamped per pair, n=4096 |
|---|---|---:|---:|---:|---:|
| accumulation | fp32 sequential | 1.89e-07 | 4.79e-07 | 1.05e-06 | 0.0 |
| accumulation | fp32 reversed | 2.35e-07 | 4.86e-07 | 1.03e-06 | 0.0 |
| accumulation | fp32 pairwise | 6.79e-08 | 8.48e-08 | 1.06e-07 | 0.0 |
| accumulation | fp32 blocked | 1.07e-07 | 1.22e-07 | 1.98e-07 | 0.0 |
| accumulation | fp16 sequential | 1.86e-03 | 3.95e-03 | 7.43e-03 | 0.0 |
| accumulation | fp16 reversed | 1.44e-03 | 3.48e-03 | 7.18e-03 | 0.0 |
| accumulation | fp16 pairwise | 5.87e-04 | 8.60e-04 | 7.77e-04 | 0.0 |
| accumulation | fp16 blocked | 6.16e-04 | 8.91e-04 | 6.94e-04 | 0.0 |
| accumulation | bf16 sequential | 1.37e-02 | 3.14e-02 | 4.96e-02 | 0.0 |
| accumulation | bf16 reversed | 1.15e-02 | 3.65e-02 | 5.89e-02 | 0.0 |
| accumulation | bf16 pairwise | 4.91e-03 | 6.56e-03 | 5.87e-03 | 0.0 |
| accumulation | bf16 blocked | 5.51e-03 | 7.60e-03 | 6.87e-03 | 0.0 |
| cast+accumulation | bf16 in / fp32 sequential | 1.74e-03 | 2.37e-03 | 2.10e-03 | 0.0 |
| cast+accumulation | bf16 in / fp32 pairwise | 1.74e-03 | 2.37e-03 | 2.10e-03 | 0.0 |
| cast+accumulation | bf16 in / fp16 sequential | 3.00e-03 | 5.69e-03 | 7.37e-03 | 0.0 |
| cast+accumulation | bf16 in / fp16 pairwise | 2.02e-03 | 2.57e-03 | 2.27e-03 | 0.0 |
| cast+accumulation | bf16 in / bf16 sequential | 1.60e-02 | 3.06e-02 | 4.70e-02 | 0.0 |
| cast+accumulation | bf16 in / bf16 pairwise | 5.11e-03 | 6.61e-03 | 6.67e-03 | 0.0 |
| scaling | e4m3 none | 3.32e-02 | 3.97e-02 | 3.80e-02 | 0.0 |
| scaling | e4m3 per-tensor | 2.88e-02 | 3.50e-02 | 3.61e-02 | 0.2 |
| scaling | e4m3 mx32 spec | 3.92e-02 | 5.13e-02 | 4.80e-02 | 57.3 |
| scaling | e4m3 mx32 +1 | 3.30e-02 | 3.98e-02 | 3.82e-02 | 0.0 |
| scaling | e2m1 none | 3.42e-01 | 4.48e-01 | 4.73e-01 | 138.3 |
| scaling | e2m1 per-tensor | 2.34e-01 | 3.54e-01 | 4.08e-01 | 0.1 |
| scaling | e2m1 mx32 spec | 1.53e-01 | 2.02e-01 | 1.97e-01 | 143.2 |
| scaling | e2m1 mx32 +1 | 2.55e-01 | 3.46e-01 | 2.92e-01 | 0.0 |
| rounding | e2m1 mx32 nearest | 1.53e-01 | 2.02e-01 | 1.97e-01 | 143.2 |
| rounding | e2m1 mx32 stochastic | 2.36e-01 | 2.79e-01 | 3.25e-01 | 143.2 |
