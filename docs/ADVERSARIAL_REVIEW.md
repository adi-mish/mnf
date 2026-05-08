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
worlds.

**Needed evidence.** Add mediation/support probes and output uncertainty over
interaction type. Real models will likely require multiple behavior metrics and
activation-local evidence.

## 5. Shared MDL depends on the coding scheme

**Attack.** Shared-MDL preference can be made to choose different atom
decompositions by changing the code language, atom granularity, or interaction
cost.

**Current response.** The theorem is now stated only relative to a fixed code.

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
evidence.

**Needed evidence.** A small real-model case study with activation hooks,
matched baselines, and intervention-prediction error. This is the point where a
GPU or dedicated model environment likely becomes necessary.

## 8. Mechanisticity score can hide arbitrary weights

**Attack.** The score

```text
Eff * exp(-(lambda_int E_int + lambda_inv E_inv + lambda_nat E_nat + beta K))
```

depends on hyperparameters. Different choices can reorder mechanisms.

**Current response.** The score is an operational scaffold, not a final
universal metric.

**Needed evidence.** Report Pareto fronts over effect, intervention error,
invariance, naturalness, and description length. Avoid relying on a single
scalar score for main claims.

## Current adversarial conclusion

The theory is internally much tighter than the initial PLAN2 sketch, but the
strong version still needs three things:

1. context-stability and abstention for interaction labels;
2. ecosystem gauge theory for fuzzy shared atoms;
3. real activation-intervention evidence against strong baselines.

The CPU-local repo can still make progress on the first two. The third likely
requires model tooling and GPU beyond tiny cases.
