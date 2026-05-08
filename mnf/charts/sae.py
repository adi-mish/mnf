from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SAETrainingResult:
    train_loss: list[float]
    l0: list[float]


class TorchTopKSAE:
    """Small optional TopK sparse autoencoder implemented in PyTorch.

    This is intentionally lightweight and not meant to compete with production
    SAE libraries.  It exists so MechanismLab-style synthetic benchmarks can
    exercise flat sparse dictionaries and compare them with typed charts.
    """

    def __init__(self, input_dim: int, latent_dim: int, k: int, seed: int = 0):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("TorchTopKSAE requires PyTorch") from exc
        if not 1 <= k <= latent_dim:
            raise ValueError("k must be in [1, latent_dim]")
        torch.manual_seed(seed)
        self.torch = torch
        self.nn = nn
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def encode_tensor(self, x):
        z = self.encoder(x).relu()
        values, indices = self.torch.topk(z, k=self.k, dim=-1)
        out = self.torch.zeros_like(z)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def reconstruct_tensor(self, x):
        return self.decoder(self.encode_tensor(x))

    def fit(self, x: np.ndarray, steps: int = 200, lr: float = 1e-3, batch_size: int = 128) -> SAETrainingResult:
        torch = self.torch
        data = torch.as_tensor(np.asarray(x, dtype=np.float32))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses: list[float] = []
        l0s: list[float] = []
        n = len(data)
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),))
            batch = data[idx]
            recon = self.reconstruct_tensor(batch)
            loss = ((recon - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, steps // 20) == 0 or step == steps - 1:
                with torch.no_grad():
                    z = self.encode_tensor(data[: min(n, 1024)])
                    losses.append(float(((self.decoder(z) - data[: min(n, 1024)]) ** 2).mean().item()))
                    l0s.append(float((z > 0).float().sum(dim=1).mean().item()))
        return SAETrainingResult(losses, l0s)

    def encode(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            z = self.encode_tensor(torch.as_tensor(np.asarray(x, dtype=np.float32)))
        return z.cpu().numpy()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            y = self.reconstruct_tensor(torch.as_tensor(np.asarray(x, dtype=np.float32)))
        return y.cpu().numpy()
