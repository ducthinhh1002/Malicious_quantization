import torch

def bit_loss(signed, margin=.1):
    """Repair all 32 bits, with additional weight on weakest quartile (8 bits).

    A proposal, not a paper reproduction or a surrogate calibrated probability.
    No relaxation of the key or of the evaluation acceptance threshold.
    """
    deficits = (margin - signed).clamp_min(0).square()
    k = max(1, signed.shape[-1] // 4)
    return .5 * deficits.mean(-1) + .5 * deficits.topk(k, dim=-1).values.mean(-1)

def pcgrad_pair(a, b):
    """Symmetric 2-task PCGrad in image space, separately for each image.

    The subsequent chain rule maps the merged image gradient to LoRA weights.
    This is not parameter-space PCGrad and gives no parameter-space guarantee.
    """
    axes = tuple(range(1, a.ndim))
    cross = (a*b).sum(axes, keepdim=True)
    aa = a.square().sum(axes, keepdim=True).clamp_min(1e-20)
    bb = b.square().sum(axes, keepdim=True).clamp_min(1e-20)
    conflict = cross.clamp_max(0)
    merged = .5 * (a-conflict/bb*b + b-conflict/aa*a)
    return merged, {'conflict_fraction': float((cross < 0).float().mean()),
                    'cosine': float((cross/(aa*bb).sqrt()).mean())}

