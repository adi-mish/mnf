# Mechanistic Normal Forms: theory sketch

For a proof-oriented version of the definitions and theorem candidates, see
`docs/FORMAL_THEORY.md`.

## 1. Object of study

Fix a model `M`, data distribution `P_X`, behavior family `B`, intervention family `I`, and error tolerance `epsilon`.  A mechanistic explanation is domain-relative.  There is no context-free, unique explanation of an entire neural network.

A mechanistic explanation is a tuple:

```text
E = (H, alpha, gamma, Omega)
```

where:

- `H` is an executable typed causal program.
- `alpha` maps low-level model states or activations into high-level program states.
- `gamma` maps high-level states/interventions back into model activation space.
- `Omega` maps model interventions to high-level interventions or vice versa.

The explanation is accepted only when it has low observation error, low intervention error, low invariance error, and low description length.

## 2. Mechanistic commutation

The core criterion is approximate intervention commutation:

```text
alpha(M^i(x)) ~= H^Omega(i)(alpha(M(x)))
```

An explanation that only predicts model outputs is behavioral.  An explanation that only labels activation directions is representational.  A mechanistic explanation predicts the effects of internal interventions.

## 3. Naturalness constraints

Unconstrained causal abstraction is too permissive.  A high-capacity nonlinear map can hide arbitrary computation inside the alignment map.  MNF therefore restricts acceptable alignments to natural maps:

- low description length,
- sparse or local use of activations,
- typed state space,
- explicit intervention semantics,
- cross-environment invariance,
- compositional reuse,
- minimality under intervention tests.

## 4. Decodable / represented / used / mechanistic

MNF separates four increasingly strong claims.

1. **Decodable:** a property can be predicted from activations by some classifier.
2. **Represented:** it can be predicted by a simple robust chart.
3. **Used:** intervening on the chart changes downstream computation as predicted.
4. **Mechanistic:** it is represented, used, invariant, minimal, and embedded in a commuting causal program.

Random-transformer SAE features illustrate the difference: labelable latents need not be learned causal mechanisms.

## 5. Typed features

A feature is not necessarily a one-dimensional direction.  A feature is a typed causal state:

```text
F = (state_space, read_map, write_map, intervention_set, downstream_role)
```

The state space may be scalar, binary, categorical, cyclic, hierarchical, relational, manifold-valued, or program-state-valued.

A weekday feature naturally has state space `C_7`, not seven unrelated scalar features and not one arbitrary direction.  A rotation intervention on weekdays is a rotation in the 2D cyclic chart, not scalar addition.

## 6. State atoms and transition atoms

Earlier feature-centric theories miss a second primitive: computation can be a sparse transition rather than a sparse state.

MNF uses two atom classes:

- **State atom:** a typed variable, such as truth, weekday phase, entity identity, refusal tendency, parent feature, child feature.
- **Transition atom:** a conditional transformation, such as an MLP map, key-value memory update, attention route, gate, or policy transform.

A residual update can be approximated as:

```text
A_{l+1} = A_l + sum_i z_i d_i + sum_j g_j(A_l) T_j A_l + error
```

SAE latents approximate state atoms.  Transcoders and MOLT-like objects approximate transition atoms.

## 7. Superposition as sparse rate-distortion

Suppose a layer has dimension `d` but there are `n > d` useful latent variables.  If variables are sparse and rarely co-active, the model can store more features than dimensions by using an overcomplete dictionary:

```text
A = sum_i z_i d_i
```

Interference is approximately:

```text
Interference_ij = P(z_i != 0, z_j != 0) <d_i, d_j>^2
```

The model solves a rate-distortion tradeoff between predictive utility, dimensional capacity, and interference.  Polysemantic neurons arise when the neuron basis is not aligned to the feature dictionary.

## 8. Feature absorption as flat-dictionary failure

If child implies parent, `C => P`, a faithful sparse code fires both parent and child on child examples.  A sparsity-optimized flat dictionary can reduce L0 by absorbing the parent feature into the child decoder:

```text
faithful: z_P = P, z_C = C
absorbed: z_P = P and not C, z_C = C
```

The parent latent then fails to fire on child cases even though the parent concept is true.  MNF predicts this whenever saved sparsity cost exceeds the reconstruction/interference penalty.

The remedy is not only larger flat SAEs.  The remedy is structured dictionaries with explicit hierarchy, typed charts, and causal consistency tests.

## 9. Circuits as minimal causal subprograms

A circuit is not a set of heads or neurons.  A circuit is a minimal subprogram:

```text
C_B subset H
```

that is:

- sufficient for behavior B,
- necessary for behavior B,
- minimal under description length,
- invariant across perturbations,
- intervention-predictive.

Heads, neurons, SAE latents, and attribution nodes are coordinate systems for approximating subprograms.

## 10. Training dynamics

For a candidate mechanism C, define training-time utility:

```text
U_t(C) = loss_reduction_t(C) - complexity(C) - interference(C) - optimization_cost(C)
```

A mechanism stabilizes when utility becomes positive and gradients coherently assemble its components.  Apparent emergence is often:

```text
continuous mechanism formation + thresholded metric + cleanup of shortcuts
```

## 11. Falsifiers

MNF is in trouble if:

- low-complexity causal programs do not predict interventions better than probes;
- random transformers pass the same mechanisticity tests as trained models;
- typed cyclic charts do not outperform scalar dictionaries on cyclic variables;
- hierarchical dictionaries do not reduce absorption;
- transition atoms do not improve MLP-heavy explanations;
- mechanism-progress metrics do not precede behavioral emergence.
