| policy | activations clamped | weights clamped | gradients clamped | attention probs clamped |
|---|---:|---:|---:|---:|
| `fp32` | 0.0% | 0.0% | 0.0% | 0.0% |
| `bf16` | 0.0% | 0.0% | 0.0% | 0.0% |
| `fp8` | 0.0% | 0.0% | 0.0% | 0.0% |
| `fp8-mx` | 0.9% | 0.8% | 0.9% | 1.9% |
| `fp8-mx-precise-probs` | 0.8% | 0.8% | 0.9% | 0.0% |
| `fp8-mx-precise-scores` | 0.9% | 0.8% | 0.9% | 1.9% |
| `fp8-mx-headroom` | 0.0% | 0.0% | 0.0% | 0.0% |
| `mxfp4` | 2.6% | 2.8% | 2.2% | 4.1% |
| `mxfp4-precise-probs` | 2.6% | 2.8% | 2.2% | 0.0% |
| `mxfp4-precise-attn` | 2.6% | 2.8% | 2.3% | 0.0% |
| `mxfp4-headroom` | 0.0% | 0.0% | 0.0% | 0.0% |
| `mxfp4-sr-bwd` | 2.8% | 2.7% | 2.5% | 9.2% |
