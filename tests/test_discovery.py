import numpy as np

from mnf.benchmarks.synthetic import sample_chain_data, make_chain_program
from mnf.core.metrics import graph_f1
from mnf.discovery.graph_discovery import discover_linear_effect_graph
from mnf.discovery.acd import AtlasCausalDiscovery


def test_discover_linear_effect_graph_on_chain():
    _, states_list = sample_chain_data(200, seed=0)
    states = {k: np.array([s[k] for s in states_list]) for k in ["x", "a", "b", "y"]}
    pred = discover_linear_effect_graph(states, threshold=0.5)
    f1 = graph_f1(make_chain_program().edges, pred)
    assert f1.recall >= 2 / 3


def test_atlas_causal_discovery_proposes_atoms():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    acd = AtlasCausalDiscovery(seed=0)
    result = acd.run(x, n_scalar_atoms=3)
    assert len(result.atoms) == 3
    assert result.program is not None
    assert "n_atoms" in result.diagnostics
