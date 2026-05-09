# Theory audit

This audit checks the current theory docs for internal consistency, proof
status, and overclaim risk. It covers:

- `docs/THEORY.md`
- `docs/FORMAL_THEORY.md`
- `docs/THEOREMS.md`
- `docs/INTERVENTIONAL_ATLASES.md`
- `docs/INTERACTION_CALCULUS.md`
- `docs/ADVERSARIAL_REVIEW.md`
- generated result summaries in `docs/RESULTS_SUMMARY.md`

## Verdict

The theory is internally consistent after demoting scalar mechanisticity in
`docs/FORMAL_THEORY.md` from a final acceptance criterion to an explicit search
Lagrangian. The current final object is the certificate-vector and atlas
formulation:

```text
A_D = (U, C, M, R, H, alpha, gamma, Omega, G, K, Q)
C(m) = (E_obs, E_int, E_inv, E_glue, E_nat,
        E_closure, K_shared, ID, effect, uncertainty, CI)
K_M(i, c) = P_M(O, A_R | x, e, c, do(i))
```

The docs now consistently treat probes, SAE latents, heads, neurons, and
directions as atom proposals rather than mechanisms unless they pass
intervention, invariance, naturalness, and certificate checks.

Mechanism identity is also now consistently response-kernel based: a fuzzy
atom-membership vector is a useful search coordinate, not the thing that makes a
mechanism real. Non-identifiability is represented explicitly by
`IdentificationSet` rather than by silently choosing one representative.

The second adversarial pass also downgraded typed ecosystem gauge
identifiability from a theorem to a conjecture/proof obligation. The cyclic
finite-state case is supported; the full ecosystem gauge case is not yet proved.

In this audit, "rigorous" means that each claim is either proved under explicit
finite/synthetic assumptions, scoped as conditional, or marked as an open
proof/evidence gap. It does not mean the current theory is globally airtight for
arbitrary learned transformer internals.

## Proof-Grade Or Nearly Proof-Grade Pieces

These claims are mathematically coherent under their stated assumptions:

- finite-table vacuity without naturalness;
- labelability without causal use;
- shortcut failure without invariance;
- pairwise factorial design economy;
- higher-order inclusion-exclusion;
- approximate contrast stability by the triangle inequality;
- context-relativity of pair labels;
- behavior-table structural ambiguity;
- context-gluing obstruction;
- response-kernel atom-splitting ambiguity;
- shared-MDL preference relative to a fixed code.

The key reason these are rigorous is that they are finite constructions or
direct inequalities. The assumptions are explicit enough to prove or falsify.

## Conditional Pieces

These are useful and consistent, but their scope must stay limited:

- hierarchy absorption is proof-grade for the simplified sparse objective, not
  for all SAE training losses;
- cyclic identifiability is proof-grade for finite cyclic variables with known
  intervention generators, not for arbitrary learned manifolds;
- transition-atom necessity is exact for piecewise affine maps with open gate
  regions, not for all MLP computation;
- path-blocking edge recovery is exact for deterministic acyclic causal
  programs with faithful interventions, not for noisy real activations;
- active design has a noiseless coverage guarantee and margin-based noisy
  stopping, not global experimental-design optimality;
- capacity competition is a principled score until a specific sparse-packing
  objective is fixed;
- ecosystem gauge identifiability is a conjecture/proof obligation beyond the
  currently implemented cyclic case;
- the tiny transformer result is learned-weight evidence, not pretrained
  real-model evidence.

## Result Consistency Checks

The headline numeric claims in the theory docs match the current generated
artifacts:

- absorption average flat parent error: `0.2247`;
- high-noise cyclic rotation error: about `0.147`;
- transition best MOLT/global MSE ratio: about `0.0001`;
- context-stability sweep changed-context rate: `0.7500`;
- feature-baseline random false-mechanism rate: `0.6667`;
- tiny transformer 20-seed mean accuracy: `0.9990`;
- tiny transformer cyclic-shift consistency: about `0.9980` for both inputs;
- tiny transformer embedding-patch consistency: about `0.9980` for both inputs;
- response-kernel atom-splitting output-only max distance: `0.0000`, with
  identification diameter `1.0000` and positive kernel distance once internal
  route markers are observed.

## Remaining Rigorous Gaps

The theory should not yet claim to solve real transformer interpretability.
The main remaining proof or evidence gaps are:

- a mature naturalness definition beyond description length, locality, and
  invariance heuristics;
- ecosystem gauge theory for fuzzy atom splitting, merging, and rotation;
- approximate/noisy response-kernel identifiability for learned models;
- real activation-intervention evidence against strong baselines such as SAE,
  ACDC, and TransformerLens workflows;
- real-model interaction evidence for redundancy, gating, support, or capacity
  competition.

The current theory is therefore best described as a coherent formal target with
finite benchmark witnesses, not a completed theory of mechanistic
interpretability.
