"""Mechanistic Normal Forms (MNF).

This package is a research scaffold for treating mechanistic interpretability as
natural causal program induction.  It intentionally contains small, testable
building blocks rather than a monolithic implementation.
"""

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.mechanism import MechanismExplanation, MNFScore
from mnf.core.types import (
    StateSpace,
    ScalarSpace,
    BinarySpace,
    CategoricalSpace,
    CyclicSpace,
    VectorSpace,
    ProductSpace,
)

__all__ = [
    "CausalProgram",
    "Node",
    "MechanismExplanation",
    "MNFScore",
    "StateSpace",
    "ScalarSpace",
    "BinarySpace",
    "CategoricalSpace",
    "CyclicSpace",
    "VectorSpace",
    "ProductSpace",
]
