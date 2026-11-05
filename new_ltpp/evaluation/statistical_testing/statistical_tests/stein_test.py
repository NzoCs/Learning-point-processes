"""One-sample goodness of fit; centered multinomial bootstrap (Yang2019 eq14).

Bootstrap validity is asymptotic; calibration must be assessed. Fitting a target
on independent data does not make that fitted target the true data-generating law.
"""

import numpy as np
import torch

from .window_tests import result


class SteinTest:
    name = "stein_papangelou"

    def __init__(self, kernel, *, n_samples=999, seed=42):
        if n_samples < 1:
            raise ValueError("Positive bootstrap count required")
        self.kernel, self.n_samples, self.seed = kernel, n_samples, seed

    def compute_statistics(self, batch):
        if len(batch.time_seqs) < 2:
            raise ValueError("At least two independent trajectories required")
        gram = self.kernel.gram(batch)
        n = len(gram)
        off = gram - torch.diag_embed(gram.diag())
        observed = off.sum() / (n * (n - 1))
        rng = np.random.default_rng(self.seed)
        weights = rng.multinomial(n, np.full(n, 1 / n), self.n_samples) - 1
        weights = torch.as_tensor(weights, device=gram.device, dtype=torch.float64)
        null = ((weights @ off) * weights).sum(1) / n**2
        return result(observed, null, "centered_multinomial_stein_bootstrap")
