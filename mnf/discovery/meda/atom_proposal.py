from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mnf.core.types import CyclicSpace
from mnf.discovery.acd import AtlasCausalDiscovery
from mnf.discovery.atoms import CandidateAtom


@dataclass(frozen=True)
class AtomProposalResult:
    atoms: tuple[CandidateAtom, ...]
    diagnostics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "atoms": [
                {
                    "name": atom.name,
                    "kind": atom.kind.value,
                    "score": atom.score,
                    "description": atom.description,
                    "metadata": dict(atom.metadata),
                }
                for atom in self.atoms
            ],
            "diagnostics": dict(self.diagnostics),
        }


def propose_atoms(
    activations: np.ndarray,
    n_scalar_atoms: int = 3,
    cyclic_labels: Sequence[Any] | None = None,
    cyclic_space: CyclicSpace | None = None,
    seed: int = 0,
) -> AtomProposalResult:
    acd = AtlasCausalDiscovery(seed=seed)
    result = acd.run(
        activations,
        n_scalar_atoms=n_scalar_atoms,
        cyclic_labels=cyclic_labels,
        cyclic_space=cyclic_space,
    )
    return AtomProposalResult(atoms=tuple(result.atoms), diagnostics=result.diagnostics)
