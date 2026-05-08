from mnf.benchmarks.synthetic import make_chain_program
from mnf.core.interventions import Intervention


def test_chain_program_runs_and_intervenes():
    prog = make_chain_program()
    state = prog.run({"x": 2.0})
    assert state["x"] == 2.0
    assert state["a"] == 5.0
    assert state["b"] == -2.25
    assert state["y"] == 5.0625
    patched = prog.run({"x": 2.0}, {"a": Intervention("a", "clamp", 0.0)})
    assert patched["a"] == 0.0
    assert patched["b"] == 0.25
    assert patched["y"] == 0.0625
    assert "x" in prog.describe()
