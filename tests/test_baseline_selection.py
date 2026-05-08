from mnf.baselines.selection import compare_shortcut_selectors


def test_invariance_selector_beats_train_only_on_shortcut_control():
    out = compare_shortcut_selectors(seed=0)
    train_only = out["train_only_selector"]
    invariant = out["invariance_selector"]
    assert train_only["selected"] == "shortcut"
    assert train_only["shifted_accuracy"] < 0.2
    assert invariant["selected"] == "causal"
    assert invariant["shifted_accuracy"] > 0.9
