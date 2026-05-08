from mnf.interactions import (
    contrast_error_bound,
    higher_order_contrast_error_bound,
    pairwise_contrast_error_bounds,
    sign_is_stable,
)


def test_pairwise_contrast_error_bounds():
    bounds = pairwise_contrast_error_bounds(0.1)
    assert bounds["joint_effect"] == 0.2
    assert bounds["synergy"] == 0.4
    assert bounds["gate_m_to_n"] == 0.4


def test_higher_order_error_bound_scales_with_cells():
    assert higher_order_contrast_error_bound(order=3, cell_error_bound=0.1) == 0.8
    assert contrast_error_bound((1, -2, 0.5), 0.2) == 0.7000000000000001


def test_sign_stability_uses_margin():
    assert sign_is_stable(1.0, 0.5)
    assert not sign_is_stable(1.0, 1.0)
    assert not sign_is_stable(1.0, 0.9, margin=0.2)
