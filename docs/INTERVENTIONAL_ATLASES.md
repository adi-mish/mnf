# Interventional mechanism atlases

PLAN3 tightens iMNF into an atlas theory. A final explanation is not a single
global circuit; it is a context-indexed family of typed causal charts with
explicit gauges, gluing errors, certificate vectors, and structural-interaction
evidence.

## Core Object

For a domain `D`, an atlas is:

```text
A_D = (U, C, M, R, H, alpha, gamma, Omega, G, K)
```

where:

- `U` is the typed atom universe;
- `C` is the chart/context cover;
- `M` is the set of fuzzy mechanism equivalence classes;
- `R` is the interaction structure;
- `H` is the executable high-level causal ecosystem;
- `alpha`, `gamma`, and `Omega` implement reads, writes, and intervention maps;
- `G` is the gauge/equivalence structure;
- `K` is the shared description-length code.

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
  U,
  CI
)
```

`mnf/certificates` implements this as `MechanismCertificate`, with Pareto
frontier and threshold-report utilities. Lower error/cost/uncertainty is better;
larger effect is better. A scalar Lagrangian is explicit and weighted.

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

## Current Boundary

The CPU-local repo now tests the atlas/certificate machinery on synthetic
systems and includes one learned-weight tiny transformer smoke test for modular
addition. It does not yet prove that the same machinery recovers mechanisms from
external pretrained transformer activations. That remains the next substantive
step before broad real-model claims.
