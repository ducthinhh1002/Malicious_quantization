"""Fixed, differentiable structural distance and per-image margin repair.

MS-SSIM: Wang, Simoncelli & Bovik (2003), five scales, Gaussian 11/1.5,
RGB channels averaged, range [0,1]. Nonpositive scale correlations are
floored at 1e-6 for finite fractional powers. This is not an anatomy metric.
"""
import torch
import torch.nn.functional as F


def ssim_components(x, y):
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 11:
        raise ValueError('Expected paired NCHW images of at least 11x11')
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, 11).expand(channels, 1, 1, 11)
    vertical = kernel.view(1, 1, 11, 1).expand(channels, 1, 11, 1)
    def blur(a):
        return F.conv2d(F.conv2d(a, horizontal, groups=channels), vertical, groups=channels)
    ux, uy = blur(x), blur(y)
    vx, vy, cov = blur(x*x)-ux*ux, blur(y*y)-uy*uy, blur(x*y)-ux*uy
    luminance = (2*ux*uy + .01**2) / (ux.square()+uy.square()+.01**2)
    contrast_structure = (2*cov + .03**2) / (vx+vy+.03**2)
    return (luminance*contrast_structure).mean((-2, -1)), contrast_structure.mean((-2, -1))


def ms_ssim(x, y):
    """Return one score per image. Inputs are RGB in [0,1], float32/float64."""
    if min(x.shape[-2:]) < 176:
        raise ValueError('Five-scale MS-SSIM needs at least 176 pixels per dimension')
    weights = x.new_tensor([.0448, .2856, .3001, .2363, .1333])
    components = []
    for level in range(5):
        similarity, contrast = ssim_components(x, y)
        components.append((similarity if level == 4 else contrast).clamp_min(1e-6))
        if level < 4:
            padding = (x.shape[-2] % 2, x.shape[-1] % 2)
            x, y = [F.avg_pool2d(z, 2, padding=padding) for z in (x, y)]
    return (torch.stack(components) ** weights[:, None, None]).prod(0).mean(1)


def margin_penalty(scores, targets, sigma):
    """Squared per-image shortfall: satisfied samples cannot cancel deficits.

    sigma is the already-frozen clean calibration standard deviation. The
    square emphasizes larger deficits; this is not a CVaR/DRO estimator.
    """
    if scores.shape != targets.shape or sigma <= 0:
        raise ValueError('Unpaired scores/targets or invalid frozen sigma')
    return ((targets.detach()-scores).clamp_min(0) / sigma).square()
