# iMNF theorem package

This document separates proof obligations from implementation details. The goal
is to make the strongest version of Interactive Mechanistic Normal Forms
falsifiable: every theorem below should either become a proof, a benchmark, or a
reason to weaken the claim.

## External anchor points

iMNF should not claim to replace causal abstraction. It should be read as a
naturalness- and interaction-constrained extension of it. Geiger et al.'s JMLR
paper frames causal abstraction as a foundation for mechanistic
interpretability and unifies activation patching, path patching, causal
mediation, circuit analysis, sparse autoencoders, masking, and steering in one
formal language:

```text
https://www.jmlr.org/papers/v26/23-0058.html
```

The iMNF addition is that intervention fit alone is not enough. The abstraction
must also be low-description-length, typed, natural, invariant, and explicit
about mechanism interactions.

The benchmark strategy should lean on three external facts:

- Tracr provides transformer models with known compiled structure:
  `https://arxiv.org/abs/2301.05062`
- ACDC is the relevant automated circuit-discovery baseline:
  `https://arxiv.org/abs/2304.14997`
- SAE evidence is mixed enough that sparse latents should be treated as atom
  proposals, not canonical ontology:
  `https://arxiv.org/abs/2501.17727`,
  `https://arxiv.org/abs/2502.04878`,
  `https://arxiv.org/abs/2602.14111`

Superposition, feature absorption, and multidimensional-feature results motivate
the iMNF terms for capacity competition, shared atoms, and typed charts:

```text
https://transformer-circuits.pub/2022/toy_model/
https://arxiv.org/abs/2409.14507
https://openreview.net/pdf?id=d63a4AM4hb
```

## Theorem A: finite-table vacuity

**Claim.** Let the observation/intervention table be finite. If `alpha`, `H`,
and `Omega` are arbitrary lookup tables and there is no naturalness or
description-length penalty, then zero observation and intervention error are
always achievable.

**Proof.** Enumerate all observed runs `(x, i)` and map each observed low-level
trace, or the activation augmented with a sample/run identifier, to a unique
high-level symbol. Let `H` return the recorded output for that symbol and let
`Omega` select the corresponding table row. This exactly fits the finite table
but stores the data rather than explaining the model. If two runs are forced to
have exactly the same low-level input to `alpha` but contradictory high-level
targets, no deterministic abstraction can fit both; the vacuity result is about
unconstrained high-capacity alignments that can memorize the empirical table.

**Repo status.** Implemented as the memorization control. This is already
proof-grade for finite data.

**Paper role.** This theorem justifies naturalness. It is the clean answer to
overly permissive causal abstraction.

## Theorem B: isolated-circuit fallacy

**Claim.** There are systems where two real mechanisms are jointly necessary
but neither is individually necessary under single ablation.

**Construction.**

```text
B_00 = 0
B_10 = 1
B_01 = 1
B_11 = 1
```

Here `m` and `n` are redundant routes. Single ablations compare `B_11` to
`B_01` or `B_10`, both zero drops. Dual ablation compares `B_11` to `B_00`, a
drop of one.

**Repo status.** `mnf/benchmarks/interactions/redundant_paths.py` and the
interaction recovery suite validate this exactly.

**Paper role.** This is the central formal break from isolated-circuit
ontology. It implies that necessity tests must be factorial or Shapley-style
when redundancy is plausible.

## Theorem C: gating can be non-identifiable from marginal interventions

**Claim.** Marginal interventions are insufficient in general for identifying
mechanisms whose role is to gate another mechanism.

**Construction.**

```text
B = gate and worker
```

The worker's effect is `1` when the gate is on and `0` when the gate is off.
On a domain where the gate is off, a marginal worker ablation has zero effect
and falsely rejects the worker. On a domain where worker prevalence changes,
the marginal gate effect is confounded by the worker distribution. The gate may
not be a direct output mechanism; it is a conditional enabler.

**Repo status.** `mnf/benchmarks/interactions/gating_mechanism.py` reports the
conditional contrast, and `mnf/interactions/factorial_effects.py` exposes
`gate_m_to_n`.

**Paper role.** This explains why attribution graphs or ablations that only ask
"does this component directly move the output?" miss control mechanisms.

## Theorem D: pairwise factorial economy

**Claim.** For `K >= 2` mechanisms, pairwise all-on-context factorial recovery
costs:

```text
1 + K + K(K - 1) / 2
```

states, versus `2^K` for full factorial recovery.

**Proof.** The unique required states are the all-on baseline, every single
ablation, and every dual ablation. For any pair `(i,j)`, these contain the four
cells needed for `Y_00`, `Y_10`, `Y_01`, and `Y_11` with all other mechanisms
held on. Deduplication gives the stated count.

**Repo status.** `mnf/interactions/design.py` implements this. The six-mechanism
recovery suite recovers all fifteen pair labels from twenty-two states rather
than sixty-four.

**Limitation.** Pairwise recovery does not identify arbitrary higher-order
interactions. The repo therefore also implements sparse higher-order
inclusion-exclusion contrasts.

## Theorem E: higher-order inclusion-exclusion

**Claim.** For a selected mechanism subset `S`, the pure `S`-order interaction
in a fixed context is:

```text
Delta_S =
  sum_{T subseteq S} (-1)^(|S| - |T|) Y_T
```

where mechanisms in `T` are on and mechanisms in `S \ T` are off.

**Proof.** This is the finite-difference operator over the Boolean cube. It
annihilates all lower-order terms and leaves the coefficient of the full
monomial over `S`.

**Repo status.** `mnf/benchmarks/interactions/higher_order.py` gives a
triple-gate contrast of `1.0`.

**Paper role.** This marks the boundary of the current pairwise engine and
gives a principled next intervention when pairwise residuals remain.

## Proposition E2: approximate contrast stability

**Claim.** Let each observed factorial cell mean have absolute error at most
`epsilon`. For any linear contrast

```text
C = sum_i c_i Y_i
```

the estimated contrast has absolute error at most:

```text
epsilon * sum_i |c_i|.
```

Therefore pairwise synergy and gate contrasts have worst-case error at most
`4 epsilon`, joint effects have error at most `2 epsilon`, and an order-`k`
inclusion-exclusion contrast has error at most `2^k epsilon`.

**Proof.** By the triangle inequality:

```text
|sum_i c_i (Y_i + e_i) - sum_i c_i Y_i|
<= sum_i |c_i| |e_i|
<= epsilon sum_i |c_i|.
```

If the true contrast magnitude is larger than this bound plus the classification
margin, its sign is stable under the cell errors.

**Repo status.** `mnf/interactions/bounds.py` implements these bounds. The noisy
recovery sweep perturbs the six-mechanism recovery suite and records the
all-correct recovery rate as noise increases. It also reports an abstention
rate using the margin-stability classifier, so weak noisy contrasts can become
`uncertain` instead of being forced into additive or non-additive labels.

**Paper role.** This is the first approximate theorem. It turns the exact
factorial claims into margin conditions for noisy intervention estimates.

## Proposition E2b: pair labels are context-relative

**Claim.** A pairwise all-on-context factorial label need not be invariant to
background interventions. There are systems where the same pair is additive
when a third mechanism is on and synergistic when that third mechanism is off.

**Construction.** Let:

```text
Y(a,b,c) = a + b + (1 - c)ab.
```

For the pair `(a,b)` with `c=1`, the synergy contrast is:

```text
Y(1,1,1) - Y(1,0,1) - Y(0,1,1) + Y(0,0,1) = 0.
```

For the same pair with `c=0`, the contrast is:

```text
Y(1,1,0) - Y(1,0,0) - Y(0,1,0) + Y(0,0,0) = 1.
```

Thus the pair label changes from additive to synergistic under a context shift.

**Repo status.** `mnf/benchmarks/interactions/context_stability.py` implements
this counterexample and reports changed labels between all-on and all-off
background contexts.

**Paper role.** This prevents overclaiming from one interaction matrix. A
serious ecosystem result should report context stability or state explicitly
which intervention context its labels describe.

## Proposition E3: active pairwise design terminates in the noiseless finite case

**Claim.** In the finite pairwise all-on-context design with noiseless
measurements, a greedy active loop that always samples an intervention state
needed by an unresolved pair terminates after at most:

```text
1 + K + K(K - 1) / 2
```

unique states for `K >= 2`, and then all pairwise factorial contrasts are
available.

**Proof.** The candidate set is exactly the deduplicated pairwise design: the
all-on baseline, every single ablation, and every dual ablation. Every
unresolved pair lacks at least one of its four cells. Sampling any missing cell
strictly reduces the number of missing cells in the finite design. After all
candidate states have been sampled, every pair has all four cells. Noiseless
contrast labels are then the exact labels induced by those four cells.

**Noisy extension.** With noisy measurements, termination is margin-based rather
than guaranteed by coverage. Repeated measurements reduce the cell confidence
radius; a pair is stable once its relevant contrast margin exceeds the
deterministic error bound. If the true margin is zero or the budget is too small,
the algorithm should report instability rather than force a label.

**Repo status.** `mnf/interactions/active.py` implements this loop, and
`mnf/benchmarks/interactions/active_design.py` reports exact recovery in `22`
states for the six-mechanism suite plus repeated-measurement behavior under
noise. The same benchmark reports matched-budget uniform and random repeated
designs as non-adaptive baselines.

## Theorem F: shared-MDL atom reuse

**Claim.** If two mechanisms can either duplicate an atom or share it, shared
MNF chooses the shared representation exactly when the shared code is shorter
than the independent code that pays duplicated atom costs:

```text
K(shared_atom) + K(m1 | shared_atom) + K(m2 | shared_atom) + K(R)
<
K_independent(m1) + K_independent(m2)
```

**Proof.** Direct comparison of the independent and shared code lengths.

**Repo status.** `mnf/mechanisms/shared_mdl.py` and
`mnf/benchmarks/interactions/shared_atom_reuse.py`.

**Paper role.** This formalizes "same concept reused in multiple mechanisms"
without forcing a binary one-feature-one-mechanism ontology.

## Proposition G: capacity competition score

**Claim.** For mechanisms with atom memberships `pi_m`, `pi_n`, atom
coactivation `q_ij`, and decoder directions `d_i`, `d_j`, a natural
first-order mechanism-level interference score is:

```text
CapComp_mn =
  sum_ij pi_mi pi_nj q_ij <d_i, d_j>^2
```

**Argument.** In sparse-packing objectives where feature interference scales
with coactivation and squared decoder alignment, mechanism memberships lift the
feature-level term to mechanism-level accounting. This is a theorem only after
the objective is fixed; otherwise it is a principled score and benchmark
quantity.

**Repo status.** `mnf/interactions/capacity.py` and the capacity phase sweep.

**Paper role.** This connects feature superposition to mechanism ecologies:
mechanisms can compete for representational capacity even when they are
separately interpretable.

## Theorem H: typed gauge identifiability

**Claim.** Mechanisms are identifiable only up to the symmetry group of their
typed state space and intervention algebra: affine transforms for scalar
charts when only affine structure is fixed, permutations for categorical
variables, rotations/reflections for cyclic variables when the origin or
orientation is not fixed, basis changes for subspaces, and atom-sharing gauge
transformations for ecosystems.

**Proof plan.** Define the intervention generators for each type and show that
commuting charts are exactly equivariant maps under the corresponding symmetry
group.

**Repo status.** The cyclic case is already implemented. The ecosystem gauge
case remains a proof task.

**Paper role.** This prevents false precision. A theory that claims canonical
neuron or SAE-latent identity where only a gauge class is identifiable is
overclaiming.

## Current stopping point

The CPU-local theory now has a coherent theorem package and executable
witnesses for the main new claims: redundancy, gating, pairwise recovery,
higher-order contrasts, approximate contrast stability, active repeated
intervention design, shared MDL, and capacity competition.

The adversarial review in `docs/ADVERSARIAL_REVIEW.md` identifies the main
remaining attack surfaces: context-dependent pair labels, additive-label
fragility under noise, non-optimal active design, redundancy/competition
confounding, shared-MDL code dependence, fuzzy-membership underidentification,
and synthetic-to-transformer transfer.

The next genuinely different step is not more synthetic proof scaffolding. It
is either:

1. a fresh mathematical pass on approximate/noisy ecosystem identifiability and
   ecosystem gauge transformations; or
2. a real-model intervention pass using transformer activations, which likely
   needs model tooling and GPU for anything beyond very small models.
