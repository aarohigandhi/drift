# GPU check

The rest of this repository simulates arithmetic and checks it against exact
rational arithmetic. That makes the simulator self-consistent. It does not make it
right about hardware.

`mx_gpu_check.ipynb` runs the three load-bearing claims against real
`torch.float8_e4m3fn` casts on a GPU:

| claim | simulator says |
|---|---|
| tanh activations clamped by the OCP shared exponent | 27–30% |
| attention probabilities clamped by the same rule | 1.9% |
| FP8 attention training with the spec exponent | diverges at steps 446, 453, 497 |
| the same with one bit of exponent headroom | trains |

Open it in Colab, pick any GPU runtime, run all cells. It takes a few minutes and
downloads nothing: the corpus is the Python language reference that ships inside
every Python install, the same text the Java runs train on.

One difference to keep in mind when reading the result. The simulator casts the
operands of the backward matmuls as well as the forward ones. The notebook casts
only the forward operands and lets autograd carry the backward pass, because that
is what a straight-through simulation on a GPU normally does. So the notebook is
the weaker test. If it still diverges, the effect is not an artifact of the Java
implementation. If it does not, that is worth knowing too, and it goes in the
README as measured.
