from mnf.benchmarks.superposition import SparseFeatureWorld, predicted_superposition_favored, superposition_phase_statistic


def test_sparse_feature_world_shapes_and_stats():
    world = SparseFeatureWorld.random(n_features=12, activation_dim=4, feature_prob=0.05, seed=0)
    z = world.sample_latents(100, seed=1)
    x = world.activations(z)
    assert z.shape == (100, 12)
    assert x.shape == (100, 4)
    stats = superposition_phase_statistic(world)
    assert stats["packing_ratio"] == 3.0
    assert stats["total_interference"] >= 0
    assert predicted_superposition_favored(12, 4, mean_coactivation=0.05**2)
