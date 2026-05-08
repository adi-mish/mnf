import math
import numpy as np

from mnf.core.types import ScalarSpace, BinarySpace, CategoricalSpace, CyclicSpace, VectorSpace, ProductSpace, Hierarchy


def test_scalar_binary_categorical_spaces():
    assert ScalarSpace(low=0, high=1).contains(0.5)
    assert not ScalarSpace(low=0, high=1).contains(2)
    assert BinarySpace().project(0.7) == 1
    cat = CategoricalSpace(["a", "b"])
    assert cat.distance("a", "a") == 0
    assert cat.distance("a", "b") == 1


def test_cyclic_space_wraparound():
    space = CyclicSpace(7, labels=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
    assert space.distance("Mon", "Sun") == 1 / 3
    assert space.project(8) == "Tue"
    emb = space.embed2("Mon")
    assert emb.shape == (2,)
    assert math.isclose(float(np.linalg.norm(emb)), 1.0)


def test_vector_product_hierarchy():
    vec = VectorSpace(2)
    assert vec.contains([1, 2])
    prod = ProductSpace({"a": ScalarSpace(), "b": vec})
    assert prod.contains({"a": 1.0, "b": np.array([0.0, 1.0])})
    h = Hierarchy({"animal": ("dog", "cat"), "dog": ("poodle",)})
    assert h.implies("poodle", "animal")
    assert "poodle" in h.descendants("animal")
