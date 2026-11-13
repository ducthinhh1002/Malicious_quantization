"""Numerical checks that never apply nonfinite gradients to optimizer state."""
import torch


def finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError('Nonfinite ' + name)


def checked_update(loss, optimizer, scaler, params):
    finite('loss', loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients:
        raise RuntimeError('No adapter gradients; check LoRA wiring/checkpointing')
    good = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
    before = scaler.get_scale()
    if not good:
        if not scaler.is_enabled():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError('Nonfinite unscaled adapter gradients')
        # GradScaler recorded overflow in unscale_: step MUST skip the update.
        # Do not clip nonfinite gradients: inf*0 would create NaNs.
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            raise RuntimeError('AMP overflow was not rejected by GradScaler')
        optimizer.zero_grad(set_to_none=True)
        return dict(skipped=1, grad_norm=None, amp_scale=scaler.get_scale(), reason='amp_overflow')
    # Accumulate norm in FP64: finite FP32 gradients can overflow an FP32 L2 sum.
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    finite('gradient norm', norm)
    coefficient = (1.0 / (norm + 1e-6)).clamp(max=1.0)
    for g in gradients:
        g.mul_(coefficient.to(device=g.device, dtype=g.dtype))
    scaler.step(optimizer)
    scaler.update()
    skipped = int(scaler.get_scale() < before)
    for p in params:
        finite('adapter parameter after optimizer update', p)
    return dict(skipped=skipped, grad_norm=float(norm), amp_scale=scaler.get_scale(), reason=None)
