# Related work, and what is not new here

This was checked after the results were in, which is the wrong order. The findings
below were reached independently, from measurement, without knowing the prior work.
That does not make them new, and this file says plainly which ones are not.

## The MX shared exponent clamps, and rounding the scale up fixes it

**Known, and published first.** Mishra, Stosic, Layton and Micikevicius,
[*Recipes for Pre-training LLMs with MXFP8*](https://arxiv.org/abs/2506.08027)
(June 2025), identify exactly this. They recommend against the OCP specification's
floor-based scale formula, because scaled values can then overflow the element
format, and instead round the shared scale factor **up** to the next power of two so
that nothing saturates. Their result is that MXFP8 with that change matches BF16 at
pre-training scale.

The `mxHeadroom` setting in this repository is the same fix, reached from the other
direction: adding one to the exponent is rounding the scale up. So the headline
"one bit of headroom rescues FP8" is a reproduction, not a discovery.

What this repository adds is the mechanism in measured form, on a model small enough
to instrument completely: the clamp rate of the tensors involved (27 to 30% of tanh
activations, 1.9% of attention probabilities), the resulting gradient shrinkage
(gain 0.900 against 1.001), and the fact that the damage tracks the *bias* of the
cast rather than its magnitude.

### The four bit case is documented too

The first version of this file listed "rounding the scale up is harmful at four
bits" as something no prior work had been found for. That was wrong, and a longer
search found it.

Yang, Deng, Nagpal, Naumov, Janani, Liu and Guan,
[*An Empirical Study of Microscaling Formats for Low-Precision LLM Training*](https://ieeexplore.ieee.org/document/11038348/)
(IEEE ARITH, 2025), compares exactly three scale rounding modes. Floor, which is
the OCP formula and which they report "often causes overflows during element
down-conversion in low-precision formats such as MXFP4". Ceil, which is rounding
the scale up, and of which they write that "although it eliminates the overflow
issues, the increased shared scale could shift more small-magnitude values to zero,
exacerbating underflow issues for smaller elements". And Even, which they propose,
rounding the scale to nearest so that neither failure dominates. They run the
comparison on Llama3 7B under MXFP4.

So the whole shape of the trade, that floor clamps and ceil underflows and neither
is right at four bits, is published, on real models, at a scale this repository
cannot reach.

Their explanation holds up when measured here. Raising the scale removes clamping
entirely, 2.59% down to 0.00%, and it raises underflow from 14.13% to 17.91% of
elements, which is the effect they name. It also cuts the bias of the cast by more
than four times and *improves* the per step gradient, gain 0.637 against 0.389. It
diverges anyway, on 8 seeds out of 8, at steps 69 to 80 on the attention model and
699 to 1442 on the MLP.

So this repository does not contradict them. It adds the observation that at four
bits the trade is not visible in per step gradient quality at all: the scale that
gives the better gradient is the one that dies.

## Bias matters more than variance in quantized training

**Known.** *Rethinking the Importance of Quantization Bias, Toward Full Low-Bit
Training* (IEEE Transactions on Image Processing, 2022) reports that the bias of
gradient quantization noise, rather than its variance, is the key factor in
accuracy loss, and that stochastic rounding does not by itself solve it. More
recently, [*Why Does Stochastic Gradient Descent Slow Down in Low-Precision
Training?*](https://arxiv.org/abs/2508.07142) (2025) analyses quantization as
magnitude shrinkage plus additive noise, and shows the shrinkage acts as a reduced
effective step size.

The measurement here, that cast bias orders the policies while cast error magnitude
does not, agrees with that literature. Per-tensor FP8 has the largest cast error of
any policy measured, 0.0348, and keeps its gradient; MX FP8 has a smaller error,
six times the bias, and loses a quarter of it.

## Attention is where low precision training breaks

**Known.** [*Why Low-Precision Transformer Training Fails: An Analysis on Flash
Attention*](https://arxiv.org/abs/2510.04212) (2025) attributes the failure to the
query against key product amplifying quantization noise and the softmax magnifying
small perturbations. Other work targets the same region from different angles, such
as suppressing activation outliers to make FP8 training stable.

The result here, that a policy which trained on an MLP diverges on attention, is
consistent with that and was not a surprise to the field.

## What has not been matched to prior work

These are not claims of novelty, only items no prior work was found for during a
short search:

- **Probes.** Computing the gradient under a different arithmetic at a run's own
  weights and batch, without touching the update, to separate "this arithmetic is
  biased" from "this arithmetic led training somewhere else". This is what showed
  that stochastic rounding is not biased per step, and that the headroom cast is not
  immune at the weights the failing run reaches.
- **The cross-language exhaustive oracle.** Every code and every ordered pair of
  codes of the 8-bit and 6-bit formats, checked against exact rational arithmetic
  written independently.
- **The observation that per step gradient quality does not predict the four bit
  outcome**: the raised scale has less bias and a better gradient and still diverges
  on every seed. Prior work names the mechanism (underflow); this adds that the
  usual per step diagnostics would have picked the wrong scale.

## What this means for how the project is described

It is an independent reproduction with a measured mechanism and a methodology for
attribution. Describing it as a discovery would be wrong, and the README and the
writeup have been changed to say so where they implied otherwise.
