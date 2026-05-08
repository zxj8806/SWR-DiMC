
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


@dataclass
class PrototypeStats:
    active: int
    usage_entropy: float
    counts: torch.Tensor


class PrototypeBank:

    def __init__(
        self,
        num_prototypes: int,
        dim: int,
        device,
        momentum: float = 0.95,
        init_from_first_batch: bool = True,
    ) -> None:
        assert 0.0 <= momentum < 1.0, "momentum must be in [0, 1)."
        self.K = int(num_prototypes)
        self.D = int(dim)
        self.device = device
        self.momentum = float(momentum)
        self.init_from_first_batch = bool(init_from_first_batch)

        self.prototypes = torch.empty(self.K, self.D, device=self.device)
        torch.nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.prototypes = l2_normalize(self.prototypes)

        self._initialized = not self.init_from_first_batch
        self.counts = torch.zeros(self.K, device=self.device)

    @torch.no_grad()
    def maybe_init(self, emb: torch.Tensor) -> None:
        if self._initialized:
            return
        emb = l2_normalize(emb)
        N = emb.size(0)
        idx = torch.arange(self.K, device=emb.device) % max(N, 1)
        self.prototypes.copy_(emb[idx])
        self._initialized = True

    @torch.no_grad()
    def assign(self, emb: torch.Tensor) -> torch.Tensor:
        self.maybe_init(emb)
        emb_n = l2_normalize(emb)
        sims = emb_n @ self.prototypes.t()
        return sims.argmax(dim=1)

    @torch.no_grad()
    def update(self, emb: torch.Tensor, idx: torch.Tensor) -> None:
        self.maybe_init(emb)
        emb_n = l2_normalize(emb)

        sums = torch.zeros_like(self.prototypes)
        counts = torch.zeros(self.K, device=self.device)
        sums.index_add_(0, idx, emb_n)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))

        mask = counts > 0
        if mask.any():
            means = sums[mask] / counts[mask].unsqueeze(1)
            means = l2_normalize(means)
            proto = self.prototypes[mask]
            proto = self.momentum * proto + (1.0 - self.momentum) * means
            self.prototypes[mask] = l2_normalize(proto)
            self.counts[mask] += counts[mask]

    @torch.no_grad()
    def stats(self) -> PrototypeStats:
        counts = self.counts.clone()
        active = int((counts > 0).sum().item())
        if counts.sum() <= 0:
            return PrototypeStats(active=active, usage_entropy=0.0, counts=counts)
        p = counts / counts.sum()
        ent = float((-p[p > 0] * torch.log(p[p > 0])).sum().item())
        return PrototypeStats(active=active, usage_entropy=ent, counts=counts)

    @torch.no_grad()
    def sample_prototypes(self, n: int) -> torch.Tensor:
        n = int(n)
        counts = self.counts
        if counts.sum() > 0:
            p = counts / counts.sum()
            idx = torch.multinomial(p, num_samples=n, replacement=True)
        else:
            idx = torch.randint(0, self.K, (n,), device=self.device)
        return self.prototypes[idx]
