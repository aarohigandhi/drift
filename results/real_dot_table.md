| policy | init: median rel. error | init: rms ratio | trained: median rel. error | trained: rms ratio |
|---|---:|---:|---:|---:|
| fp32 sequential | 6.80e-07 | 7.93e-07 | 4.70e-07 | 6.46e-07 |
| fp32 pairwise | 9.22e-08 | 9.36e-08 | 5.63e-08 | 6.09e-08 |
| fp32 blocked | 1.63e-07 | 1.76e-07 | 1.02e-07 | 1.27e-07 |
| fp16 sequential | 5.51e-03 | 6.65e-03 | 3.87e-03 | 5.17e-03 |
| fp16 pairwise | 7.72e-04 | 7.68e-04 | 4.58e-04 | 4.87e-04 |
| fp16 blocked | 8.30e-04 | 8.50e-04 | 3.73e-04 | 4.00e-04 |
| bf16 sequential | 4.27e-02 | 5.06e-02 | 2.98e-02 | 3.94e-02 |
| bf16 pairwise | 5.87e-03 | 6.02e-03 | 3.65e-03 | 4.00e-03 |
| bf16 blocked | 6.49e-03 | 6.65e-03 | 2.94e-03 | 3.17e-03 |
| bf16 cast | 2.42e-03 | 2.42e-03 | 7.36e-04 | 7.86e-04 |
| e4m3 per-tensor cast | 3.81e-02 | 3.83e-02 | 1.27e-02 | 1.32e-02 |
| e4m3 mx32 cast | 4.36e-02 | 4.27e-02 | 1.42e-02 | 1.47e-02 |
