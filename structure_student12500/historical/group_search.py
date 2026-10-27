"""Small, auditable adaptations of attribution and Pareto-grid search.

Not a reproduction of KernelSHAP-NAS or MPaGE's LLM heuristic synthesis.
No test metrics enter these functions.
"""
import itertools
import math
import re
import numpy as np


def group_for_key(key):
    if '.lora_A.' not in key and '.lora_B.' not in key:
        raise ValueError(f'Unexpected adapter tensor {key}')
    module = re.split(r'\.lora_[AB]\.', key)[0]
    region = next((x for x in ('down', 'mid', 'up') if f'{x}_block' in key), None)
    # Boundary convolutions belong to the adjacent down/up region, so no
    # trained tensor is silently left at the original strength.
    if module.rsplit('.',1)[-1]=='conv_in': region='down'
    if module.rsplit('.',1)[-1]=='conv_out': region='up'
    if region is None:
        raise ValueError(f'Unmapped adapter region: {key}')
    if any(x in module for x in ('.to_q', '.to_k', '.to_v', '.to_out.')):
        kind = 'attention'
    elif 'conv' in module:
        kind = 'conv'
    else:
        raise ValueError(f'Unmapped adapter target: {key}')
    return region + '_' + kind


def scale_state(state, groups, vector):
    import torch
    if len(groups) != len(vector) or len(set(groups)) != len(groups):
        raise ValueError('Invalid group vector')
    if any(not math.isfinite(float(v)) or not 0 < v <= 1 for v in vector):
        raise ValueError('Scales must be finite and in (0, 1]')
    mapping = dict(zip(groups, vector))
    actual = {group_for_key(k) for k in state}
    if actual != set(groups):
        raise ValueError('Group coverage mismatch')
    out = {}
    for k, v in state.items():
        partner = k.replace('.lora_A.', '.lora_B.') if '.lora_A.' in k else k.replace('.lora_B.', '.lora_A.')
        if partner not in state or not torch.isfinite(v).all():
            raise ValueError('Missing partner or nonfinite adapter')
        out[k] = (v.float() * mapping[group_for_key(k)] if '.lora_B.' in k else v.float()).contiguous()
    return out


def coalitions(n):
    if not 1 <= n <= 6:
        raise ValueError('Predeclared experiment supports at most six groups')
    return list(itertools.product((0, 1), repeat=n))


def kernel_shap(masks, values):
    """Kernel-weighted least squares with exact empty/full endpoint constraints.

    Complete coalition enumeration removes coalition-sampling variance; image
    sampling uncertainty still remains. For full enumeration this equals the
    Shapley allocation, including interactions.
    """
    x = np.asarray(masks, dtype=float); y = np.asarray(values, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError('Invalid attribution inputs')
    n = x.shape[1]
    if sorted(map(tuple, x)) != sorted(coalitions(n)):
        raise ValueError('Expected each binary coalition exactly once')
    counts = x.sum(1).astype(int)
    base = float(y[counts == 0][0]); total = float(y[counts == n][0] - base)
    if n == 1:
        phi = np.array([total])
    else:
        keep = (counts > 0) & (counts < n)
        z = x[keep]; k = counts[keep]
        w = np.array([(n-1)/(math.comb(n, int(s))*s*(n-s)) for s in k])
        # Eliminate last coefficient to impose sum(phi) == full - empty.
        design = z[:, :-1] - z[:, -1:]
        target = y[keep] - base - z[:, -1]*total
        a = design * np.sqrt(w[:, None]); b = target * np.sqrt(w)
        if np.linalg.matrix_rank(a) != n-1:
            raise ValueError('Unidentifiable attribution')
        prefix = np.linalg.lstsq(a, b, rcond=None)[0]
        phi = np.append(prefix, total-prefix.sum())
    return {'empty': base, 'full': base+total, 'phi': phi.tolist(),
            'interaction_residual_rmse': float(np.sqrt(np.mean((base+x@phi-y)**2))),
            'coalitions': len(x), 'method': 'endpoint-constrained KernelSHAP WLS, complete coalition panel'}


def attribution_vector(attributions, n):
    """Predeclared sensitivity heuristic, not an estimated optimal watermark."""
    def norm(x):
        a = np.asarray(x, float)
        return a / max(float(np.linalg.norm(a, ord=1)), 1e-12)
    merit = (norm(attributions['margin_p10']['phi'])
             - norm(attributions['lpips']['phi'])
             - .25*norm(attributions['clip_drop']['phi']))
    high = sorted(range(n), key=lambda i: (-merit[i], i))[:max(1, n//2)]
    return [0.85 if i in high else 0.65 for i in range(n)], merit.tolist()


def objectives(row):
    v = [row['lpips'], row['clip_drop'], -row['margin_p10']]
    if not all(math.isfinite(x) for x in v):
        raise ValueError('Nonfinite objective')
    return np.array(v)


def pareto_front(rows):
    scores = [objectives(r) for r in rows]
    return [r for i, r in enumerate(rows) if not any(
        np.all(s <= scores[i]) and np.any(s < scores[i]) for j, s in enumerate(scores) if i != j)]


def grid_elites(rows, bins=4):
    front = pareto_front(rows)
    if not front:
        raise ValueError('Empty archive')
    z = np.stack([objectives(r) for r in front])
    z = (z-z.min(0))/np.maximum(z.max(0)-z.min(0), 1e-12)
    cells = {}
    for row, pos in zip(front, z):
        cell = tuple(np.minimum((pos*bins).astype(int), bins-1))
        cells.setdefault(cell, []).append(row)
    return [sorted(cells[k], key=lambda r: r['tag']) for k in sorted(cells)]


def proposal(rows, seen, n, seed, mode='pareto'):
    rng = np.random.default_rng(seed)
    levels = np.array([.65, .70, .75, .85, 1.])
    cells = grid_elites(rows) if mode == 'pareto' else None
    for attempt in range(2000):
        if mode == 'random' or attempt > 100:
            vector = rng.choice(levels, n).tolist()
        else:
            def parent():
                cell = cells[int(rng.integers(len(cells)))]
                return cell[int(rng.integers(len(cell)))]['vector']
            left, right = parent(), parent()
            vector = [left[i] if rng.random() < .5 else right[i] for i in range(n)]
            j = int(rng.integers(n)); vector[j] = float(rng.choice(levels))
        key = tuple(vector)
        if key not in seen:
            return vector
    raise RuntimeError('Search space exhausted')


def shortlist(rows, count=3):
    """Quality, margin and compromise anchors; no test data or quality cutoff."""
    front = pareto_front(rows)
    z = np.stack([objectives(r) for r in front])
    z = (z-z.min(0))/np.maximum(z.max(0)-z.min(0), 1e-12)
    candidates = [min(front, key=lambda r: (r['lpips'], r['tag'])),
                  max(front, key=lambda r: (r['margin_p10'], r['tag'])),
                  front[int(np.argmin(np.linalg.norm(z, axis=1)))]]
    out = []
    for r in candidates + sorted(front, key=lambda r: r['lpips']):
        if r['tag'] not in [v['tag'] for v in out]:
            out.append(r)
        if len(out) >= count:
            break
    return out
