from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.core.dsl import MechanismDSL
from mnf.core.validation import validate_node_set


def test_mechanism_dsl_from_program():
    prog = make_chain_program()
    dsl = MechanismDSL.from_program(prog)
    text = dsl.to_text()
    assert "mechanism chain_x_a_b_y" in text
    assert "x" in dsl.variables()
    assert "outputs: y" in text


def test_validate_node_set_on_chain():
    prog = make_chain_program()
    inputs, _ = sample_chain_data(32, seed=0)
    val = validate_node_set(prog, inputs, keep_nodes=["x", "a", "b", "y"])
    assert val.sufficiency_error == 0.0
    val2 = validate_node_set(prog, inputs, keep_nodes=["a"])
    assert val2.necessity_effect > 0.0
