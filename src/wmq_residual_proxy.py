"""Fixed TRAIN residual subspaces for calibration; no ownership identification.

Uncentered SVD retains shared residual directions. SELECT never fits or updates
the basis or loss normalization. Gaussian subspace is a same-rank control.
"""
import torch


class ResidualProxy:
    def __init__(self, residuals, rank=2, random=False, seed=0, orthogonal_weight=.25):
        matrix = torch.cat(residuals).detach().float().flatten(1)
        if not torch.isfinite(matrix).all() or rank < 1 or orthogonal_weight < 0:
            raise ValueError('Invalid residual proxy input')
        # Few calibration rows: use CPU SVD to avoid a large CUDA workspace.
        _, singular, vt = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        effective = int((singular > max(float(singular[0])*1e-5, 1e-8)).sum())
        count = min(rank, effective)
        basis = vt[:count]
        if random and count:
            generator = torch.Generator().manual_seed(seed)
            basis = torch.linalg.qr(torch.randn(matrix.shape[1], count, generator=generator), mode='reduced').Q.T
        self.basis = basis.to(matrix.device)
        self.width = matrix.shape[1]
        self.energy = matrix.square().mean().clamp_min(1e-8).detach()
        self.orthogonal_weight = orthogonal_weight
        self.diagnostics = {'rank': count, 'requested_rank': rank, 'active': bool(count),
                            'random_basis': random, 'fit_residual_rms': float(matrix.square().mean().sqrt()),
                            'normalization_energy': float(self.energy), 'normalization_source': 'FIT only',
                            'fit_capture_fraction': self.capture(matrix)}

    def capture(self, residuals):
        matrix = residuals.flatten(1).float()
        return float(((matrix @ self.basis.T).square().sum()/matrix.square().sum().clamp_min(1e-12)).clamp(0, 1))

    def loss(self, displacement, residual):
        d, r = displacement.float().flatten(1), residual.float().flatten(1)
        projected_d = d @ self.basis.T
        projected_r = r @ self.basis.T
        matching = (projected_d-projected_r).square().sum(1)/self.width
        outside = (d.square().sum(1)-projected_d.square().sum(1)).clamp_min(0)/self.width
        return ((matching+self.orthogonal_weight*outside)/self.energy).mean()
