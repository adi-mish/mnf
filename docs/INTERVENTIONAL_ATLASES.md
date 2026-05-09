# Interventional mechanism atlases

PLAN4 tightens iMNF into a response-kernel atlas theory. A final explanation is
not a single global circuit; it is a context-indexed family of typed causal
charts with explicit gauges, gluing errors, identification sets, certificate
vectors, and structural-interaction evidence.

The current target can be read as:

```text
Mechanistic interpretability is recovery of a minimal natural factorization
of a model's interventional response kernel.
```

For fixed inputs, environments, contexts, interventions, and observables, the
model induces a finite or sampled response kernel:

```text
K_M(i, c) = P_M(O, A_R | x, e, c, do(i)).
```

A mechanism claim is accepted only insofar as its high-level atlas predicts this
kernel under the available interventions and reports which alternative
factorizations remain indistinguishable.

## Core Object

For a domain `D`, an atlas is:

```text
A_D = (U, C, M, R, H, alpha, gamma, Omega, G, K, Q)
```

where:

- `U` is the typed atom universe;
- `C` is the chart/context cover;
- `M` is the set of fuzzy mechanism equivalence classes;
- `R` is the interaction structure;
- `H` is the executable high-level causal ecosystem;
- `alpha`, `gamma`, and `Omega` implement reads, writes, and intervention maps;
- `G` is the gauge/equivalence structure;
- `K` is the shared description-length code;
- `Q` is the uncertainty and identification certificate.

The local acceptance condition is intervention commutation:

```text
alpha_c(M^i(x, e)) ~= H_c^{Omega_c(i)}(alpha_c(M(x, e)))
```

with naturalness, invariance, uncertainty, and shared-MDL constraints.

## Certificate Vectors

The repo now treats scalar mechanisticity as a search heuristic. Final claims
should report a certificate vector:

```text
C(m) = (
  E_obs,
  E_int,
  E_inv,
  E_glue,
  E_nat,
  E_closure,
  K_shared,
  ID,
  U,
  CI
)
```

`mnf/certificates` implements this as `MechanismCertificate`, with Pareto
frontier and threshold-report utilities. Lower error/cost/uncertainty is better;
larger effect is better. `ID` is an `IdentificationSet`: the set of alternative
representatives that the current intervention/observable algebra cannot
distinguish. A scalar Lagrangian is explicit and weighted.

`mnf/semantics` implements finite response kernels and kernel-distance helpers.
`mnf/benchmarks/identifiability.py` includes a deliberately small
atom-splitting ambiguity: output-only observations cannot distinguish a split
redundant route from a merged route, while adding internal route markers makes
the alternatives distinguishable.

## Gluing

Charts are allowed to use different coordinates. On overlap, they must agree up
to an allowed gauge transform:

```text
epsilon_glue(c, d) = E[d(psi_{c->d}(alpha_c(A)), alpha_d(A))]
```

`mnf/atlas` implements `MechanismChart`, `GaugeTransform`, and `gluing_error`.
`mnf/benchmarks/atlas/no_global_chart.py` shows a minimal obstruction: no single
global gauge can glue two contexts, while a context-indexed gauge has zero
error.

## Structural Interaction Claims

Pairwise output tables establish phenomenological interactions only. The same
four-cell output table can arise from a directed gate or symmetric synergy.
Therefore structural labels require internal evidence.

`mnf/interactions/structural.py` implements this separation:

```text
phenomenological_label = synergistic_or_gated
structural_label = ambiguous_without_internal_evidence
```

`mnf/benchmarks/interactions/table_aliasing.py` then adds internal evidence that
orients a directed gate while leaving behavior-only evidence ambiguous.

## Mechanism Identity

Mechanisms are not identical to fuzzy membership vectors. A mechanism is an
intervention-stable response factor, considered up to the gauge and
non-identifiability structure of the chosen domain. Fuzzy atom membership is a
useful implementation coordinate for search, packing, and shared-MDL accounting,
but final claims should be about response-kernel behavior plus a certificate.

## Current Boundary

The CPU-local repo now tests the atlas/certificate machinery on synthetic
systems, explicitly reports one response-kernel non-identifiability witness, and
includes one learned-weight tiny transformer smoke test for modular addition
with embedding and final-token block activation patching. It also includes a
learned two-route tiny transformer where single ablations underweight redundant
routes and dual ablation reveals the missing causal mass. It does not yet prove
that the same machinery recovers hidden mechanisms from external pretrained
transformer activations. That remains the next substantive step before broad
real-model claims.
