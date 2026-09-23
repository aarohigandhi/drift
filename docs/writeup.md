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

So the honest state is that attention is more fragile under every low precision setting I tried, that one bit of headroom is the difference between a working FP8 run and a dead one, and that the mechanism behind the failure is still open. I would rather publish the open question with the two suspects ruled out than a tidy explanation I cannot support.

The code, every result and every retraction are at [github.com/aarohigandhi/drift](https://github.com/aarohigandhi/drift).
