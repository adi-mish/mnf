from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from mnf.charts.linear import LinearChart
from mnf.charts.cyclic import CyclicChart
from mnf.core.causal_program import CausalProgram, Node
from mnf.core.types import ScalarSpace, CyclicSpace, VectorSpace
from mnf.discovery.atoms import CandidateAtom, AtomKind
from mnf.discovery.graph_discovery import discover_linear_effect_graph, edge_effect_scores, prune_edges_by_mdl


@dataclass
class DiscoveryResult:
    atoms: list[CandidateAtom]
    edges: set[tuple[str, str]]
    program: CausalProgram | None
    diagnostics: dict[str, Any]


class AtlasCausalDiscovery:
    """Prototype discovery pipeline for small synthetic MNF benchmarks.

    The method is intentionally conservative: it only proposes charts and graphs
    that can be scored and tested.  Real LLM usage would replace the simple
    activation arrays with TransformerLens hooks, SAE/crosscoder features, and
    activation-patching validators.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def propose_scalar_atoms(
        self,
        activations: np.ndarray,
        n_atoms: int,
        prefix: str = "pc",
    ) -> tuple[list[CandidateAtom], LinearChart]:
        chart = LinearChart.fit(activations, latent_dim=n_atoms, name=f"{prefix}_linear_chart")
        z = chart.encode(activations)
        atoms = [
            CandidateAtom(
                name=f"{prefix}_{i}",
                kind=AtomKind.STATE,
                activations=z[:, i],
                score=float(np.var(z[:, i])),
                description="principal linear state candidate",
            )
            for i in range(n_atoms)
        ]
        return atoms, chart

    def propose_cyclic_atom(
        self,
        activations: np.ndarray,
        labels: Sequence[Any],
        space: CyclicSpace,
        name: str = "cyclic",
    ) -> tuple[CandidateAtom, CyclicChart]:
        chart = CyclicChart.fit(activations, labels, space)
        xy = chart.encode_xy(activations)
        err = chart.intervention_error_after_rotation(activations, labels, steps=1)
        atom = CandidateAtom(
            name=name,
            kind=AtomKind.STATE,
            activations=xy,
            score=1.0 - err,
            description=f"typed cyclic state C_{space.period}",
            metadata={"period": space.period, "rotation_error": err},
        )
        return atom, chart

    def discover_graph_from_atoms(
        self,
        atoms: Sequence[CandidateAtom],
        threshold: float = 0.2,
        penalty: float = 0.05,
    ) -> set[tuple[str, str]]:
        states = {a.name: np.asarray(a.activations) for a in atoms if a.activations is not None}
        edges = discover_linear_effect_graph(states, threshold=threshold)
        scores = edge_effect_scores(states, edges)
        return prune_edges_by_mdl(edges, scores, penalty=penalty)

    def fit_linear_program_from_states(
        self,
        states: Mapping[str, np.ndarray],
        edges: set[tuple[str, str]],
        outputs: Sequence[str],
    ) -> CausalProgram:
        """Create a simple linear structural equation program from ordered states.

        Each node equation predicts the node by linear regression on its parents.
        Source nodes read from inputs directly.
        """
        names = list(states.keys())
        nodes: list[Node] = []
        coefs: dict[str, tuple[list[str], np.ndarray, float]] = {}
        for name in names:
            parents = [s for s, d in edges if d == name and s in names[: names.index(name)]]
            y = np.asarray(states[name], dtype=float).reshape(len(states[name]), -1)
            if y.shape[1] != 1:
                # For vector-valued states, use norm as a scalar in this tiny program.
                y = np.linalg.norm(y, axis=1, keepdims=True)
            if not parents:
                coefs[name] = ([], np.zeros((0, 1)), float(y.mean()))

                def make_source(n: str):
                    return lambda _state, inputs, n=n: float(inputs.get(n, coefs[n][2]))

                nodes.append(Node(name, (), ScalarSpace(), make_source(name), description="source state"))
            else:
                X = np.column_stack([np.asarray(states[p], dtype=float).reshape(len(states[p]), -1)[:, 0] for p in parents])
                X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
                beta = np.linalg.lstsq(X_aug, y[:, 0], rcond=None)[0]
                coefs[name] = (parents, beta[:-1], float(beta[-1]))

                def make_func(n: str):
                    def func(state, _inputs, n=n):
                        ps, w, b = coefs[n]
                        return float(sum(w[i] * state[p] for i, p in enumerate(ps)) + b)
                    return func

                nodes.append(Node(name, tuple(parents), ScalarSpace(), make_func(name), description="linear effect equation"))
        return CausalProgram(nodes, tuple(outputs), name="discovered_linear_program")

    def run(
        self,
        activations: np.ndarray,
        n_scalar_atoms: int = 3,
        cyclic_labels: Sequence[Any] | None = None,
        cyclic_space: CyclicSpace | None = None,
    ) -> DiscoveryResult:
        atoms, _chart = self.propose_scalar_atoms(activations, n_scalar_atoms)
        if cyclic_labels is not None and cyclic_space is not None:
            atom, _cchart = self.propose_cyclic_atom(activations, cyclic_labels, cyclic_space)
            atoms.append(atom)
        edges = self.discover_graph_from_atoms(atoms)
        states = {a.name: a.activations for a in atoms if a.activations is not None and np.asarray(a.activations).ndim == 1}
        program = None
        if states:
            program = self.fit_linear_program_from_states(states, edges, outputs=[list(states.keys())[-1]])
        diagnostics = {"n_atoms": len(atoms), "n_edges": len(edges), "atom_scores": {a.name: a.score for a in atoms}}
        return DiscoveryResult(atoms=list(atoms), edges=edges, program=program, diagnostics=diagnostics)
