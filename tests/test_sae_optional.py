import importlib.util
import numpy as np


def test_optional_torch_sae_tiny_smoke():
    if importlib.util.find_spec("torch") is None:
        return
    from mnf.charts.sae import TorchTopKSAE

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 4)).astype("float32")
    sae = TorchTopKSAE(input_dim=4, latent_dim=8, k=2, seed=0)
    result = sae.fit(x, steps=3, lr=1e-2, batch_size=16)
    z = sae.encode(x[:5])
    recon = sae.reconstruct(x[:5])
    assert z.shape == (5, 8)
    assert recon.shape == (5, 4)
    assert len(result.train_loss) >= 1
