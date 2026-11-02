"""Project the *actual AdamW displacement*, not its preconditioner-free gradient.

Small exact active-set QP for <=3 homogeneous first-order descent constraints.
This is an experimental PCGrad-inspired variant, not PCGrad/CAGrad reproduction.
It provides no nonlinear-loss, full-gradient, or downstream detection guarantee.
"""
import itertools
import torch


def dot(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


def norm(a):
    return float(dot(a, a).clamp_min(0).sqrt())


def project_displacement(delta, gradients):
    """Closest d to delta with <g_i,d> <= 0. Unit normals, FP64 tiny QP.

    Enumerating all active sets handles dependent/opposed gradients. Origin
    is always feasible, but is used only if it is the nearest candidate.
    """
    if len(gradients) > 3:
        raise ValueError('At most three constraints supported')
    if any(not torch.isfinite(x).all() for x in delta):
        raise RuntimeError('Nonfinite proposed update')
    normals = []
    for gradient in gradients:
        length = norm(gradient)
        if not torch.isfinite(torch.tensor(length)):
            raise RuntimeError('Nonfinite protected gradient')
        if length > 1e-12:
            normals.append([g / length for g in gradient])
    dn = norm(delta)
    if not normals or dn == 0:
        return [d.clone() for d in delta], {'active': [], 'norm_before': dn, 'norm_after': dn}
    # Work in unit displacement coordinates so the tolerances are independent
    # of learning rate. Reconstruct in parameter dtype only after solving.
    gram = torch.tensor([[float(dot(a, b)) for b in normals] for a in normals], dtype=torch.float64)
    b = torch.tensor([float(dot(g, delta)) / dn for g in normals], dtype=torch.float64)
    candidates = []
    for size in range(len(normals) + 1):
        for active in itertools.combinations(range(len(normals)), size):
            lam = torch.zeros(len(normals), dtype=torch.float64)
            if active:
                idx = list(active)
                sub = gram[idx][:, idx]
                answer = torch.linalg.pinv(sub, rtol=1e-12) @ b[idx]
                if (answer < -1e-9).any() or not torch.allclose(sub @ answer, b[idx], atol=1e-8, rtol=1e-8):
                    continue
                lam[idx] = answer.clamp_min(0)
            if (b - gram @ lam <= 1e-8).all():
                candidates.append((float(lam @ gram @ lam), active, lam))
    if not candidates:
        raise RuntimeError('Descent-cone QP has no numerical solution')
    _, active, lam = min(candidates, key=lambda x: x[0])
    result = [d.clone() for d in delta]
    for coefficient, gradient in zip(lam.tolist(), normals):
        for out, g in zip(result, gradient):
            out.add_(g, alpha=-coefficient * dn)
    residual = [float(dot(g, result)) / dn for g in normals]
    if max(residual, default=0.) > 2e-6:
        raise RuntimeError('Projection violated first-order constraints')
    return result, {'active': list(active), 'norm_before': dn, 'norm_after': norm(result),
                    'normalized_dot_before': b.tolist(), 'normalized_dot_after': residual}


class ProtectedAdamW(torch.optim.AdamW):
    """Keep Adam moments as proposal state; project its realized displacement.

    Moments are NOT the moments of the projected displacement. Both are saved
    by the ordinary optimizer checkpoint; constraints are refreshed each batch.
    """
    def __init__(self, params, *, project=True, **kwargs):
        super().__init__(params, **kwargs)
        self.project = project
        self.constraints = []
        self.last_projection = {}

    @torch.no_grad()
    def step(self, closure=None):
        params = [p for group in self.param_groups for p in group['params']]
        if not self.constraints:
            self.last_projection = {'applied': False, 'auxiliary_step': False}
            return super().step(closure)
        before = [p.clone() for p in params]
        result = super().step(closure)
        delta = [p - old for p, old in zip(params, before)]
        proposed_dots = [float(dot(g, delta)) for g in self.constraints]
        if self.project:
            delta, stats = project_displacement(delta, self.constraints)
            for p, old, d in zip(params, before, delta):
                p.copy_(old + d)
        else:
            stats = {'norm_before': norm(delta), 'norm_after': norm(delta)}
        # Quantization when adding FP32 parameters can perturb an exact cone
        # boundary; report the actually stored step as well as the QP result.
        actual = [p - old for p, old in zip(params, before)]
        self.last_projection = {**stats, 'applied': self.project, 'auxiliary_step': True,
            'proposal_dots': proposed_dots, 'realized_dots': [float(dot(g, actual)) for g in self.constraints],
            'realized_norm': norm(actual),
            'constraint_order': ['original_loss', 'negative_native_score', 'native_quality'],
            'scope': 'local truncated-gradient linearization only; not a detection guarantee'}
        self.constraints = []
        return result
