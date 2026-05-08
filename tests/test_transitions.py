import numpy as np

from mnf.charts.transitions import fit_linear_transition, MixtureOfLinearTransforms


def test_linear_transition_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    W = rng.normal(size=(3, 2))
    b = rng.normal(size=2)
    y = x @ W + b
    atom = fit_linear_transition(x, y)
    pred = atom(x)
    assert np.mean((pred - y) ** 2) < 1e-10


def test_mixture_of_linear_transforms_runs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 2))
    y = np.zeros((300, 1))
    mask = x[:, 0] > 0
    y[mask, 0] = 2 * x[mask, 0] + 1
    y[~mask, 0] = -3 * x[~mask, 1] - 1
    model = MixtureOfLinearTransforms.fit_with_kmeans_gates(x, y, n_atoms=2, seed=0)
    pred = model.predict(x)
    assert pred.shape == y.shape
    assert np.mean((pred - y) ** 2) < 3.0  # hard kmeans gates are approximate
