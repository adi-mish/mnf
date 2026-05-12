from mnf.benchmarks import benchmark_card_index


def test_benchmark_cards_include_plan4_metadata():
    cards = benchmark_card_index()
    card = cards["atom_splitting_identifiability"]

    data = card.as_dict()

    assert data["known_nonidentifiabilities"]
    assert data["allowed_interventions"]
    assert data["expected_certificate_thresholds"]
