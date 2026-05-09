import pytest

from mnf.benchmarks.identifiability import atom_splitting_identifiability_metrics
from mnf.semantics import (
    InterventionAlgebra,
    estimate_response_kernel,
    identification_set_from_kernels,
    kernel_max_distance,
)
from mnf.semantics.response_kernel import ResponseKernel


def test_response_kernel_distance_detects_shared_support_difference():
    a = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 1.0)
    b = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 0.25)

    assert kernel_max_distance(a, b) == 0.75


def test_identification_set_from_indistinguishable_kernels():
    a = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 1.0)
    b = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 1.0)
    identification = identification_set_from_kernels({"a": a, "b": b}, reference="a")

    assert identification.representatives == ("a", "b")
    assert identification.is_ambiguous is True


def test_identification_set_can_report_structural_diameter():
    a = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 1.0)
    b = estimate_response_kernel(("i0",), ("c",), ("out",), lambda _i, _c, _o: 1.0)
    identification = identification_set_from_kernels({"a": a, "b": b}, reference="a", diameter=2.0)

    assert identification.representatives == ("a", "b")
    assert identification.diameter == 2.0


def test_atom_splitting_identifiability_reports_needed_evidence():
    metrics = atom_splitting_identifiability_metrics()

    assert metrics["output_only_max_distance"] == 0.0
    assert metrics["output_only_identification"]["is_ambiguous"] is True
    assert metrics["output_only_identification"]["diameter"] == 1.0
    assert metrics["rich_observable_max_distance"] > 0.0
    assert metrics["rich_observables_distinguish"] is True


def test_response_kernel_distance_requires_identical_support():
    a = ResponseKernel({("none", "default", "output"): 1.0})
    b = ResponseKernel(
        {
            ("none", "default", "output"): 1.0,
            ("none", "default", "marker"): 0.0,
        }
    )

    with pytest.raises(ValueError, match="identical support"):
        kernel_max_distance(a, b)


def test_intervention_algebra_composition_and_closure():
    algebra = InterventionAlgebra(
        interventions=("none", "ablate_a", "ablate_b", "ablate_both"),
        composition={
            ("ablate_a", "ablate_a"): "ablate_a",
            ("ablate_b", "ablate_b"): "ablate_b",
            ("ablate_a", "ablate_b"): "ablate_both",
            ("ablate_b", "ablate_a"): "ablate_both",
            ("ablate_a", "ablate_both"): "ablate_both",
            ("ablate_both", "ablate_a"): "ablate_both",
            ("ablate_b", "ablate_both"): "ablate_both",
            ("ablate_both", "ablate_b"): "ablate_both",
            ("ablate_both", "ablate_both"): "ablate_both",
        },
    )

    assert algebra.compose("none", "ablate_a") == "ablate_a"
    assert algebra.compose("ablate_a", "ablate_b") == "ablate_both"
    assert algebra.is_closed
