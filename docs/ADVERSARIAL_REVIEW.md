# Adversarial theory review

This document argues against the current iMNF theory. Its purpose is to keep
the repo honest: every objection should either become a theorem condition, a
benchmark, or a reason to weaken the claim.

## 1. Factorial recovery can overfit the chosen context

**Attack.** Pairwise all-on-context recovery estimates interactions only when
all non-pair mechanisms are held on. A pair can look additive in that context
and synergistic when a third mechanism is off.

**Current response.** The docs now say "selected context" rather than claiming
context-free recovery. The higher-order inclusion-exclusion benchmark covers one
triple-gate case, and the context-stability benchmark gives an explicit pair
whose label changes across background interventions.

**Needed evidence.** Extend the context-shifted benchmark from one analytic
counterexample into a sweep over background mechanisms and report interaction
stability across environments, not just two contexts.

## 2. Additive labels are fragile under noise

**Attack.** If additive means "no detectable interaction", the label is only as
good as the tolerance. With a tiny tolerance, noise creates false interactions;
with a large tolerance, weak real interactions disappear.

**Current response.** `mnf/interactions/bounds.py` gives deterministic contrast
bounds and the noisy recovery sweep uses the bound as the classification
tolerance. The abstaining classifier reports `uncertain` when the selected label
is not stable to a modestly wider error bar.

**Needed evidence.** Calibrate abstention with bootstrap intervals and repeated
interventions in real activation settings, not only synthetic cell noise.

## 3. Active design is not globally optimal

**Attack.** The active loop is a greedy repeated-measurement rule. It is not an
optimal experimental design, and it can waste samples if the uncertainty proxy
is poorly calibrated.

**Current response.** The theory now claims only finite noiseless coverage and
margin-based noisy stopping, not optimality. The active-design benchmark now
includes uniform and random repeated-design baselines at matched budgets.

**Needed evidence.** Add richer design families and statistical efficiency
curves. The current baselines test the obvious alternatives, not global
optimality.

## 4. Redundancy and competition can be confounded

**Attack.** Negative synergy can mean redundancy, saturation, ceiling effects,
or genuine destructive interference. A four-cell behavioral table alone may not
separate these mechanisms.

**Current response.** The current classifier uses separate redundancy and
competition scores, but the distinction is heuristic outside clean synthetic
worlds. The table-aliasing benchmark now makes this limitation explicit:
behavior-only evidence can be structurally ambiguous, and directed structural
labels require internal evidence.

**Needed evidence.** Add mediation/support probes and output uncertainty over
interaction type. Real models will likely require multiple behavior metrics and
activation-local evidence.

## 5. Shared MDL depends on the coding scheme

**Attack.** Shared-MDL preference can be made to choose different atom
decompositions by changing the code language, atom granularity, or interaction
cost.

**Current response.** The theorem is now stated only relative to a fixed code.
The atlas module now makes gauge/gluing an explicit object, including a minimal
case where no single global gauge is adequate.

**Needed evidence.** Prove gauge-invariant comparisons or report sensitivity to
reasonable code families.

## 6. Fuzzy mechanism membership is underidentified

**Attack.** Many atom-membership vectors can produce the same behavior and
shared description length. Without additional constraints, fuzzy membership may
be a useful parameterization rather than an identifiable object.

**Current response.** The repo validates overlap and shared-MDL accounting but
does not yet prove identifiability.

**Needed evidence.** Develop ecosystem gauge theory: characterize when two
different atom-membership decompositions are equivalent under atom splitting,
merging, or rotation.

## 7. Synthetic mechanisms may not transfer to transformers

**Attack.** The current suite uses executable toy mechanisms. Real transformer
interventions are approximate, entangled, distribution-shifting operations.

**Current response.** The docs explicitly stop short of claiming real-model
evidence. The repo now includes 100-seed learned tiny-transformer CPU sweeps:
one modular-addition transformer with high cyclic and activation-patching
consistency, and one two-route redundant modular transformer where single
ablations underweight routes while dual ablation collapses behavior.

**Needed evidence.** A small real-model case study with activation hooks,
matched baselines, and intervention-prediction error. This is the point where a
GPU or dedicated model environment likely becomes necessary.

## 7b. The redundant tiny transformer may be too explicit

**Attack.** The redundant tiny transformer has two separate route modules by
construction. It is therefore not evidence that an arbitrary transformer's
hidden redundant routes can be discovered without architectural hints.

**Current response.** The result is scoped as learned-weight evidence for the
redundancy/certificate logic, not as hidden-route discovery. It does show the
main isolated-circuit failure under training: either route alone remains
sufficient, so a single-ablation baseline underweights the routes, while dual
ablation reveals the causal mass.

**Needed evidence.** Train a less explicitly separated transformer where
redundant algorithms share the same residual stream, then recover the routes
from activation-local atoms rather than from named route modules.

## 7c. Activation-patching consistency can be task-position trivial

**Attack.** For modular addition, patching the final readout token from a
shifted source run may simply transplant the answer-bearing state. This is
valid causal evidence, but it is weaker than decomposing attention q/k/v or MLP
submodules into interpretable mechanisms.

**Current response.** The benchmark now distinguishes shallow embedding patches
from later final-token block patches and reports both. It does not claim q/k/v
or MLP-local recovery.

**Needed evidence.** Add hook-level decomposition for attention patterns,
attention output, MLP preactivation, and MLP output, with negative controls
where patching the wrong site or wrong token fails.

## 8. Mechanisticity score can hide arbitrary weights

**Attack.** The score

```text
Eff * exp(-(lambda_int E_int + lambda_inv E_inv + lambda_nat E_nat + beta K))
```

depends on hyperparameters. Different choices can reorder mechanisms.

**Current response.** The score is an operational scaffold, not a final
universal metric. `mnf/certificates` now exposes certificate vectors,
thresholds, and Pareto fronts so scalar Lagrangians are only explicit search
utilities.

**Needed evidence.** Report Pareto fronts over effect, intervention error,
invariance, naturalness, and description length. Avoid relying on a single
scalar score for main claims.

## Current adversarial conclusion

The theory is internally much tighter than the initial PLAN2 sketch, but the
strong version still needs three things:

1. broader context-stability and abstention calibration beyond the current sweep;
2. ecosystem gauge theory for fuzzy shared atoms beyond the minimal atlas-glue benchmark;
3. hidden-route recovery and real activation-intervention evidence against
   strong baselines.

The CPU-local repo can still make progress on the first two. The third likely
requires model tooling and GPU beyond tiny cases.
