"""A training-channel surrogate, not a guarantee of downstream radioactivity."""
import math
import torch


def learning_rate_factor(step, plateau=10000, total=15000, minimum=.2):
    if not 0 <= step or not 0 < plateau < total or not 0 < minimum <= 1:
        raise ValueError('Invalid learning-rate schedule')
    progress = min(1., max(0., (step - plateau) / (total - plateau)))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def robust_margin(scores, targets, sigma):
    """N x views: half mean + half worst-view squared normalized deficit.

    No averaging images before applying the hinge. This is NOT population CVaR.
    The maximum over three views is an explicit finite-view surrogate.
    """
    if scores.ndim != 2 or targets.shape != scores.shape or sigma <= 0:
        raise ValueError('Expected paired N x views scores/targets and positive sigma')
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise ValueError('Nonfinite score/target')
    losses = ((targets.detach() - scores).clamp_min(0) / sigma).square()
    return .5 * losses.mean(1) + .5 * losses.max(1).values


def bit_margin(coordinates, signs, floor=.25):
    # Coordinate axes already standardized by the frozen carrier.
    return (floor - coordinates * signs).clamp_min(0).square().mean(-1)
