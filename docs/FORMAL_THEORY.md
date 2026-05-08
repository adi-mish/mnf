# Mechanistic Normal Forms: formal theory draft

This document is a proof-oriented draft. It is not a completed theory of
mechanistic interpretability. Its purpose is to make the MNF claims precise
enough that they can be proved, falsified, or turned into benchmark tests.

## 1. Objects

Fix a domain

```text
D = (P_X, B, I_L, E, epsilon)
```

where `P_X` is an input distribution, `B` is a behavior or metric family, `I_L`
is a family of low-level interventions, `E` is a set of environments, and
`epsilon` is the allowed error tolerance.

A low-level model is a deterministic or stochastic map

```text
M : X -> A -> Y
```

where `A` denotes internal states or activations. A high-level explanation is a
tuple

```text
MNF = (H, alpha, gamma, Omega)
```

where:

- `H` is an executable typed causal program over variables `Z_1, ..., Z_k`.
- `alpha : A -> Z` reads high-level state from low-level activations.
- `gamma : Z -> A` or `gamma : interventions(Z) -> interventions(A)` writes
  high-level states or interventions back into low-level coordinates.
- `Omega : I_L -> I_H` maps low-level interventions to high-level interventions,
  or maps high-level interventions to low-level implementations.

The core commutation condition is:

```text
alpha(M^i(x)) ~= H^Omega(i)(alpha(M(x)))
```

for `x ~ P_X` and `i in I_L`.

## 2. MNF objective

An explanation is evaluated by

```text
L_MNF =
  E_obs
  + lambda E_int
  + mu E_inv
  + beta K(MNF)
  + rho E_unnatural
```

where:

- `E_obs` is behavior prediction error.
- `E_int` is intervention prediction error.
- `E_inv` is cross-environment invariance failure.
- `K(MNF)` is description length.
- `E_unnatural` penalizes nonlocal reads, high-capacity alignments, memorizing
  maps, missing intervention semantics, and brittle prompt- or environment-
  specific explanations.

The important point is that `E_obs` alone is not an interpretability objective.
Observation fit can be achieved by shortcuts or lookup tables.

## 3. Levels of claim

For a property `Y` and activation `A`:

1. **Decodable:** some predictor `q` has low prediction error for `Y`.
2. **Represented:** a simple, robust chart `q in Q_simple` predicts `Y`.
3. **Used:** interventions on the chart change downstream computation in the
   predicted way.
4. **Mechanistic:** the variable is represented, used, invariant, minimal, and
   embedded in a commuting causal program.

MNF is mainly a theory of the fourth level. It treats probes, SAE latents,
heads, neurons, and directions as coordinate systems unless they pass the
intervention and invariance tests.

## 4. Theorem candidates

### Theorem 1: vacuity without naturalness

**Claim.** Let `X` and `I_L` be finite. If `alpha` and `H` may be arbitrary
lookup tables with no description-length or naturalness penalty, then for any
finite observation/intervention table there exists an MNF tuple with zero
observation error and zero intervention error.

**Proof sketch.** Enumerate every observed pair `(x, i)` and assign it a unique
symbol `s_(x,i)`. Let `alpha` map the low-level activation observed under
intervention `i` on input `x` to `s_(x,i)`. Let `H` be a table that returns the
recorded high-level output for every enumerated symbol. Let `Omega` map each
low-level intervention to the corresponding table index. This construction fits
the finite table exactly. It reveals no mechanism because all computation is
hidden in the alignment/table. Its description length grows with the number of
observed cases.

**Benchmark link.** `mnf/benchmarks/memorization.py` implements this pathology:
the memorizing alignment gets train accuracy `1.0`, test accuracy about `0.475`,
and description length `512.0`.

**Status.** This is the cleanest formal argument in the theory: naturalness is
not optional. Without it, causal abstraction is non-identifying.

### Theorem 2: shortcut selectors fail without invariance

**Setup.** There are two candidate variables, causal `C` and shortcut `S`, and
two environments, train `e_0` and shifted `e_1`. The label is stable under `C`.
The shortcut has higher train accuracy but lower shifted accuracy:

```text
Acc_train(S) > Acc_train(C)
Acc_shift(C) > Acc_shift(S)
gap(S) = |Acc_train(S) - Acc_shift(S)| is large
gap(C) is small
```

**Claim.** A train-only labelability selector chooses `S`. An invariance-aware
selector that requires high minimum environment accuracy and small environment
gap chooses `C`.

**Proof sketch.** The train-only selector maximizes `Acc_train`, so it chooses
`S` by assumption. The invariance selector first filters candidates with
`min_e Acc_e >= a_min` and `max_e Acc_e - min_e Acc_e <= g_max`. Choose
thresholds such that `C` satisfies both constraints and `S` violates the gap
constraint. Then `S` is rejected and `C` is selected.

**Benchmark link.** `mnf/baselines/selection.py` creates a hard shortcut case:
the train-only selector chooses `shortcut` with train accuracy `0.9805` and
shifted accuracy `0.1075`; the invariance selector chooses `causal` with shifted
accuracy `0.9635`.

**Status.** This is a proposition, not a deep theorem, but it is central. It
formalizes why labelability is weaker than mechanisticity.

### Theorem 3: hierarchy absorption boundary

**Setup.** Let `C => P`, where `C` is a child feature and `P` is a parent
feature. A faithful flat sparse code uses:

```text
z_P = P
z_C = C
```

An absorbed code uses:

```text
z_P = P and not C
z_C = C
```

Let `p_C = Pr(C = 1)`. Let `lambda` be the sparsity penalty. Let
`DeltaRecon = Recon_absorbed - Recon_faithful`, and let `DeltaInterference` be
the extra interference cost of absorption.

**Claim.** The absorbed code has lower objective than the faithful flat code iff

```text
lambda p_C > DeltaRecon + DeltaInterference.
```

**Proof sketch.** The faithful code pays parent sparsity cost on all parent
cases, including child cases. The absorbed code saves exactly one parent firing
on each child case. Its saved sparsity cost is `lambda p_C`. Absorption is
favored exactly when the saved sparsity cost exceeds the added reconstruction
and interference costs.

**Benchmark link.** `mnf/experiments/run_absorption_phase.py` sweeps this
failure mode. In the current generated results, average flat parent error is
about `0.2247`; hierarchy-aware recovery has error `0.0`.

**Status.** The inequality is proof-grade for the simplified objective. The
open work is proving analogous results for realistic SAE losses and learned
dictionaries.

### Theorem 4: cyclic variables are identifiable only up to gauge

**Setup.** Let the true high-level variable live in `C_n`, with generator
intervention `tau(k) = k + 1 mod n`. Let a chart be an injective map
`phi : C_n -> A` with a decoder that recovers the cyclic state. Suppose a
candidate chart `psi` has zero observation error and commutes with the generator
intervention.

**Claim.** If the orientation of `tau` is known, `psi` is identifiable up to a
rotation of `C_n`. If only adjacency/cyclic distance is known, it is identifiable
up to rotation and reflection.

**Proof sketch.** Pick the image of one state, say `0`. Commutation with `tau`
forces the image of every other state:

```text
psi(k) = psi(tau^k(0)).
```

Thus the choice of `psi(0)` determines a rotation. If orientation is not fixed,
the inverse generator `tau^-1` gives an equally valid chart, giving reflection.
No scalar ordering is canonical.

**Benchmark link.** `mnf/benchmarks/cyclic.py` and
`mnf/experiments/run_cyclic_noise_sweep.py` instantiate this with weekdays.
Rotation-intervention error is `0.0` through noise `0.08`, and about `0.147` at
noise `0.3`.

**Status.** This is proof-grade for finite cyclic variables with known
intervention generators. It should generalize to compact group variables, but
that requires more machinery.

### Theorem 5: transition atoms are necessary for short faithful descriptions

**Setup.** Let a mechanism be a gated piecewise-linear map

```text
Y = W_g X + b_g,  g in {1, ..., K}.
```

A transition-atom explanation stores the gate and `K` local transforms. A global
linear explanation stores one transform `W`.

**Claim.** If there exist gates `g != h` such that `W_g != W_h` or `b_g != b_h`
on a region with positive probability, no single global linear map is exactly
faithful. A transition-atom explanation can be exact with description length
roughly proportional to `K` local transforms plus the gate. A state-only
explanation must either accept intervention error or encode the conditional
transform in expanded state variables.

**Proof sketch.** If a single `W,b` were exact, then for all `x` in each gate
region, `Wx+b = W_gx+b_g`. On any region with nonempty interior, equality of
affine maps implies `W = W_g` and `b = b_g`. If two gate regions have different
affine maps, contradiction. The transition-atom representation directly stores
the true conditional maps.

**Benchmark link.** `mnf/benchmarks/transition_only.py` instantiates this. In
the current sweep, at gate separation `2.5` and noise `0.05`, the MOLT/global
MSE ratio is about `0.021`; at gate separation `4.0` and no noise, it is about
`0.0001`.

**Status.** Exact for piecewise affine maps with open gate regions. The open
work is measuring the same distinction in real MLP layers or transcoder-style
replacements.

### Theorem 6: path-blocking interventions identify direct edges in acyclic programs

**Setup.** Let `H` be an acyclic deterministic structural causal program with
ordered variables. For a candidate edge `u -> v`, define a path-blocking test:
run a baseline input, clamp every predecessor of `v` except `u` to its baseline
value, counterfactually change `u`, then measure whether `v` changes.

**Claim.** Under faithful interventions and nondegenerate structural equations,
the path-blocking test rejects transitive-only edges and accepts direct parent
edges.

**Proof sketch.** If every path from `u` to `v` passes through an intermediate
predecessor, clamping all non-`u` predecessors of `v` to baseline blocks the
effect, so `v` does not change. If `u` is a direct parent and `v`'s structural
equation depends nontrivially on `u`, changing `u` while holding other parents
fixed changes `v` on a positive-measure set.

**Benchmark link.** `mnf/core/validation.py` implements this. The ground-truth
suite recovers direct graphs with F1 `1.0` for scalar chain, gated XOR, modular
addition, and relational lookup.

**Status.** Proof-grade for deterministic finite synthetic programs. Real model
activation patching is the analogue, but it needs approximate/noisy versions of
the assumptions.

### Proposition 7: thresholded metrics can make gradual mechanisms look sudden

**Setup.** Let a mechanism-progress variable `m_t` increase smoothly during
training. Let a behavioral metric be a thresholded readout

```text
b_t = sigma(k (m_t - theta))
```

with large sharpness `k`.

**Claim.** Mechanistic progress can cross an intervention-predictive threshold
strictly before the behavioral metric crosses its emergence threshold, even
though both are monotone functions of the same underlying mechanism.

**Proof sketch.** For `m_t` increasing, the crossing time for any threshold is
the first `t` with `m_t >= threshold`. If the mechanistic threshold is lower
than the behavioral threshold `theta` needed to make `b_t` large, the mechanism
crossing occurs earlier. Large `k` makes `b_t` appear abrupt.

**Benchmark link.** `mnf/benchmarks/training_dynamics.py` instantiates this
with a smooth mechanism trace and a thresholded behavioral metric.

**Status.** This is a toy proposition, but it makes the training-dynamics claim
explicit and testable.

## 5. What this theory explains

The current MNF formalization explains why several common interpretability
signals are insufficient:

- High train accuracy can come from shortcuts.
- Feature labelability can exist without causal use.
- Observation fit can be produced by memorizing alignments.
- Scalar charts can be unnatural for cyclic variables.
- State-only explanations can be long or inaccurate for transition-heavy
  mechanisms.
- Correlation graphs over-propose transitive dependencies without intervention
  validation.

## 6. What remains open

The theory is not complete. The largest missing pieces are:

1. A mature definition of naturalness beyond simple description-length and
   invariance penalties.
2. Approximate versions of the theorems for noisy learned models.
3. A proof that MNF-style scores are identifiable under realistic intervention
   families.
4. Real-model evidence against strong baselines such as SAE, ACDC, and
   TransformerLens workflows.
5. A training-dynamics result connecting mechanism formation to later behavior.

The current repo should therefore be read as a formal target plus a falsifiable
benchmark scaffold, not as a completed theory of mechanistic interpretability.

## 7. Relation to current literature

MNF is closest in spirit to causal abstraction. Geiger et al.'s JMLR paper
argues that causal abstraction can provide a common formal language for
mechanistic interpretability methods including activation patching, path
patching, causal mediation, causal scrubbing, causal tracing, circuit analysis,
concept erasure, sparse autoencoders, masking, and steering [1]. MNF accepts
that starting point but adds a stronger naturalness requirement: an abstraction
that hides computation inside a high-capacity alignment map is not yet a
mechanistic explanation.

The need for naturalness is sharpened by the non-linear representation dilemma:
if arbitrary nonlinear maps are allowed, causal abstraction can become too
permissive [2]. The vacuity theorem above is the finite-table version of this
concern.

MNF also treats current feature-discovery tools as measurement coordinates
rather than final ontology. OpenAI's SAE scaling work trained a 16-million
latent autoencoder on GPT-4 activations and showed SAE quality metrics improving
with scale [3]. That is strong evidence that sparse coordinates are useful.
But random-transformer SAE results show that labelable SAE latents can appear
even in randomly initialized transformers [4], so labelability is not sufficient
for mechanism. Feature absorption results show another failure mode: sparse
latents can look monosemantic while failing to fire where they semantically
should [5]. MNF's response is to require causal use, hierarchy consistency,
invariance, and intervention commutation.

Typed variables are motivated by work on non-linear and multidimensional
features. Engels et al. define irreducible multidimensional features and report
automatically discovered circular features for weekdays and months [6]. MNF
therefore treats a direction as a chart, not as the ontology.

The benchmark side follows the Tracr lesson: compiled or otherwise
ground-truth mechanisms are valuable laboratories for interpretability [7].
The graph-validation side is related to ACDC, which uses activation-patching-
based pruning to discover circuit structure [8]. MNF's local path-blocking
validator is a finite synthetic analogue of those intervention tests.

ROME motivates the transition-atom view: factual associations can sometimes be
localized and edited through low-rank updates to feed-forward modules [9]. MNF
rephrases this as a special case where a localized transition atom is an
adequate explanation; it predicts failures when facts are distributed,
context-conditioned, or entangled with hierarchy.

## References

[1]: https://www.jmlr.org/papers/v26/23-0058.html "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"

[2]: https://arxiv.org/abs/2507.08802 "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?"

[3]: https://arxiv.org/abs/2406.04093 "Scaling and evaluating sparse autoencoders"

[4]: https://arxiv.org/abs/2501.17727 "Sparse Autoencoders Can Interpret Randomly Initialized Transformers"

[5]: https://arxiv.org/abs/2409.14507 "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"

[6]: https://arxiv.org/abs/2405.14860 "Not All Language Model Features Are Linear"

[7]: https://arxiv.org/abs/2301.05062 "Tracr: Compiled Transformers as a Laboratory for Interpretability"

[8]: https://arxiv.org/abs/2304.14997 "Towards Automated Circuit Discovery for Mechanistic Interpretability"

[9]: https://arxiv.org/abs/2202.05262 "Locating and Editing Factual Associations in GPT"
