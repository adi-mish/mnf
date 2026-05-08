import numpy as np

from mnf.benchmarks.induction import (
    compare_induction_vs_memorization,
    induction_copy_predict,
    make_induction_generalization_dataset,
    make_induction_program,
)
from mnf.experiments.run_induction_demo import run


def test_induction_copy_predicts_held_out_pairs():
    ds = make_induction_generalization_dataset(seed=0)
    pred = induction_copy_predict(ds.sequences)
    assert np.mean(pred == ds.labels) == 1.0


def test_induction_program_runs_match_copy():
    prog = make_induction_program(vocab_size=16)
    state = prog.run({"sequence": [2, 5, 11, 2, 5]})
    assert state["match_position"] == 0
    assert state["copied_token"] == 11
    assert state["y"] == 11


def test_induction_mechanism_generalizes_better_than_memorizer():
    out = compare_induction_vs_memorization(seed=0)
    assert out["memorizer_train_accuracy"] == 1.0
    assert out["copy_test_accuracy"] == 1.0
    assert out["memorizer_test_accuracy"] < 0.25


def test_induction_demo_smoke():
    out = run(seeds=(0, 1))
    assert out["mean_copy_test_accuracy"] == 1.0
    assert out["mean_memorizer_test_accuracy"] < 0.25
