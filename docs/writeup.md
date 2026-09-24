# What actually breaks low precision training

Every large model today trains on smaller numbers than computers were built around. Weights and activations get squeezed into 16, 8, sometimes 4 bits, and when a run goes wrong people reach for a familiar list of suspects. Not enough exponent bits. A stale scaling factor. The wrong rounding mode.

What bothered me is that those explanations almost never get measured. On a GPU they can't be. The number format, the width of the running sum, the order the products are added in and the way tensors are scaled all come baked into one kernel. You can't change one of them and hold the rest still.

In software you can. So I built drift. It does the arithmetic of a matrix multiply bit for bit in Java, turns each of those decisions into its own setting, and trains real models under every one.

## Making the arithmetic trustworthy first

None of it means anything if the simulated formats are wrong, so the first thing I built was a way to catch myself. Every format is written twice with no shared code. Java rounds the fast way. Python redoes everything in exact fractions, with no floating point anywhere. A script then compares the two on every value each format can hold, and on every possible pair of values as a product and as a sum. For the 8 bit formats that is all 65,536 pairs, not a sample.

It found 154 disagreements in the E4M3 table on its first run. Every one was negative zero. A positive zero times a negative number is negative zero, these formats spend a bit pattern on it, and a Python fraction has no sign on zero. Java was right and my reference was wrong. I liked that the check caught its own author on day one.

## The order you add things in

Floating point addition isn't associative, so the order a kernel sums its products can change the answer. The question is whether that matters.

Mostly it doesn't, until the running sum is narrow. In a 32 bit sum the error stays around a millionth whatever the order. In bf16, adding 4,096 products one after another loses about 5%, while adding them as a balanced tree loses about 0.6%. At that length the sequential 16 bit sum loses more than casting the inputs all the way down to 8 bits, which costs 3.6%.

Synthetic vectors can flatter a result like that, so I checked it on real tensors from the 2,048 wide first layer of a trained model. The gap grew. After training, the 8 bit cast lost 1.3% while the sequential 16 bit sum still lost 3%.

Then the question I actually cared about: does it change training? In a model built for it, summation order made a 6× difference to the error in the gradient, 10% against 1.6%. Across 8 seeds, though, the final loss showed nothing I could tell apart from noise. With 3 seeds I had written that a 16 bit accumulator "costs something". With 8, that didn't hold, so I took it back.

## A rule in the spec that clamps

The MX formats scale each block of 32 numbers by a shared power of two, chosen by a formula in the OCP specification. That formula places the largest value of the block in a range the element format can't fully reach. FP8 stops at 448 in a range that runs to 512, so the biggest values in a block get clipped.

In training this cost more than I expected. With MX scaled FP8, every gradient came out about 10% too short. The cause was specific and measurable: 27 to 30% of activations were being clamped. The model uses tanh, whose outputs pile up just under 1.0, which is exactly the worst place for that rule. One extra bit of exponent headroom brought the gradient back to full length and the loss back to where 32 bit training was.

For 4 bit numbers the same fix did the opposite. Training with the extra bit diverged on all three seeds.

## Why stochastic rounding hurt

Stochastic rounding is supposed to be the careful option, because each rounding is unbiased on average. In 4 bit training it was the worst setting that still trained.

Splitting it in half showed that all of the damage came from rounding the gradients, not the weights or activations. The harder question was why. Comparing two runs can't answer that, because after a few hundred steps they sit at different weights, and any gap could be the arithmetic or simply where training ended up.

So I built probes. At every measurement a probe computes the gradient another arithmetic would have produced, at the run's own weights and on the same batch, without touching the update. A test checks that a probed run is identical to an unprobed one.

The answer was clean. At the same weights, ordinary and stochastic rounding shrank the gradient by the same amount, within 0.006. Stochastic rounding wasn't biased at any single step. What it did was steer training toward weights where 4 bit casts clamp more, 52% of activations instead of 37%, and at those weights either kind of rounding kept only about half the gradient. A cast that never clamps, probed at those same weights, kept 99% of it through step 1,000.

That probe turned up something I didn't expect. At the weights of the standard run, the cast that never clamps gives a gradient that is nearly unbiased and about as accurate as the standard one. Train with it and it diverges anyway. How good a gradient looks at one point didn't predict what the arithmetic would do to a whole run. It also meant an explanation I had already put in the README, that at 4 bits resolution matters more than clamping, wasn't supported. I removed it.

## What I take from it

The format is rarely the whole story. The things that actually moved training here were a scaling rule meeting the place activations happen to sit, and the path a run takes rather than the error at any one step. Neither shows up if you only compare final losses.

The limits are real. The models are small, the largest has 148k parameters, and everything runs on a CPU.

## Attention, and a prediction I got wrong

Everything above ran on models with no attention, so I built one: a single block of causal attention followed by an MLP, with the same measurements pointed at it. Two of its matmuls have no equivalent in the earlier models. A score is a query against a key, and a context is softmax probabilities against values, so in both cases two activations multiply and cast error arrives on both sides.

My prediction was specific. Softmax probabilities sit just under 1.0, the same place tanh outputs sit, so I expected them to clamp hard under the MX rule and to explain whatever went wrong. That was wrong twice over.

It was wrong about the clamping. Under MX scaled FP8, about 1.9% of attention probabilities clamp, against 27 to 30% of the tanh activations in the MLP. A tenth as much.

It was wrong about the cause. The same MX scaled FP8 that trained fine on the MLP now blows up on every seed, at steps 446, 453 and 497. So a tenth of the clamping, and the run dies rather than surviving. One extra bit of exponent headroom still rescues it, which keeps the scaling rule firmly in the story, but the story is not the one I wrote down in advance.

Then I checked the obvious suspects directly. I added settings that keep the score matmul, or the probability matmul, in full precision while every ordinary layer stays cast. Both variants diverge too, at steps 439 to 484, which is indistinguishable from leaving them cast. Whatever kills attention under MX scaled FP8, it is not the casting of the attention operands. I do not know yet what it is, and the repository says so.

What I can measure is where the gradient goes short. In attention the worst shrinkage is at the query, key and value projections, and the mildest is at the output, which is the reverse of the MLP, where shrinkage grew with depth. At 4 bits attention is simply harder: it costs 0.36 of held out loss against 0.10 on the MLP, and gradients keep 40% of their length against 71%. Here the attention matmuls do carry part of the blame, since holding both in full precision recovers about a quarter of the gap.

So I looked at the failure itself. It is a runaway rather than one bad step: the loss sits near 2.8 at step 430, reaches 5.2 by 440, and is gone at 446, while the gradient has been decaying for about seventy five steps before that.

Then I probed the dying run with other casts at its own weights, which is the trick that worked earlier. Over its last sixty steps the run keeps 76% of its gradient. The same weights under a cast with one extra bit of headroom keep 89%, so that cast suffers there too. The same weights under ordinary FP8 with one scale factor per tensor keep 102%, which is to say nothing at all goes wrong. Meanwhile the clamp rate sits flat at about 1% through the whole collapse, and slightly falls.

That rules out the story I would have told. Clamping is not the trigger, because it never rises. The failure belongs to block scaling rather than to 8 bit numbers, because per tensor scaling at the very same weights is untouched. And the extra bit of headroom does not work by being immune, since at the bad weights it degrades as well. It works by never arriving there: along its own healthy run the spec cast probes steady, and nothing collapses.

So I kept measuring instead of guessing. Underflow, meaning values that fall to zero because a block scale was set by something much larger, stays near nothing. The relative error of the cast itself is flat. The spread of values inside one scaling block barely moves. None of them tracks a gradient falling from 0.92 to 0.76.

What does track it is the bias of the cast. Not how large the error is, but which way it points. I measured the projection of the cast error back onto the tensor it came from: zero would mean the error is sideways noise, and negative means the cast quietly shrinks whatever it touches, which is what clipping the biggest element of every block would do.

Lined up across four settings, bias predicts the gradient and error size does not. Ordinary FP8 with one scale per tensor has the largest cast error of anything I ran, and it keeps essentially all of its gradient. MX scaled FP8 has a smaller error and loses a quarter. One extra bit of exponent headroom uses the very same block scaling, clips nothing, removes about fourteen fifteenths of the bias, and with it all of the damage. Four bit MX has the most bias of all and keeps only 41% of its gradient.

That is a satisfying shape for an answer. A one way error shrinks a gradient. An error thirty times bigger that points in no particular direction does not.

One piece is still open and I want to name it exactly. Between settings, bias predicts the gradient. Within the dying run it does not: the bias is flat while the gradient collapses and the run blows up. So the bias explains why MX scaled FP8 is worse everywhere, and not what tips this particular run over at step 430. No quantity I measured moves when it does. That one stays in the repository as an open question rather than a story.

## What of this was already known

I checked the literature after the fact, which is the wrong order, and it cost me the
headline. The scaling fix is published. NVIDIA's recipe for pre training with MXFP8,
from June 2025, says not to use the floor based scale from the specification, because
scaled values then overflow the format, and says to round the shared scale up instead.
Adding one to the exponent, which is what I did, is that same fix arrived at from the
other side. So the strongest result in this project is a reproduction.

The bias result has precedent too. Work from 2022 reports that the bias of quantization
noise, rather than its variance, is what costs accuracy, and a 2025 analysis treats
quantization as magnitude shrinkage that behaves like a smaller step size. That is the
same shape as what I measured. And attention being the fragile part of a low precision
run is itself an active topic, with a 2025 paper attributing it to the query against key
product amplifying noise and the softmax magnifying it.

So what is left that is mine. The measurements, which are careful and which nobody owed
me: clamp rates, gradient gain against an exact gradient, and bias separated from error
magnitude across four settings. The probes, which compute what a different arithmetic
would have done at a run's own weights, and which is how I separated a biased cast from
a bad trajectory. The exhaustive oracle across every code and every pair of codes. And
one case that runs against the published recipe: at four bits, rounding the scale up
made every run diverge here, where the specification's scale trained.

I would rather say that clearly than let a reader assume I found something first. The
work is an independent reproduction with the mechanism measured and a method for
attribution, and [docs/related_work.md](https://github.com/aarohigandhi/drift/blob/main/docs/related_work.md)
in the repository lays out line by line which is which.

The code, every result and every retraction are at [github.com/aarohigandhi/drift](https://github.com/aarohigandhi/drift).
