"""Owner-blind genetic search over hard W4 scales and rounding thresholds.

Spatial targets are unverified proxies, not watermark estimates. No hidden
trigger, extractor, key, or TEST inputs are accepted by this module.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from wmq_evolution import search, ranked
from wmq_grouped_quant import GroupedW4
from wmq_sleeper_equivariance import guided_spatial_target, predicted_x0


class GeneticWeight(nn.Module):
    def __init__(self, weight, group_size, initialization):
        super().__init__()
        grid = GroupedW4(weight, group_size, initialization)
        self.register_buffer('base_scale', grid.base_scale)
        self.register_buffer('cached', weight.detach().clone())
        self.width, self.padding, self.shape = grid.width, grid.padding, grid.shape
        self.enabled = False

    @torch.no_grad()
    def set_gene(self, original, gene):
        scale = self.base_scale * float(np.exp(.04*gene[0]))
        blocks = F.pad(original.detach().reshape(self.shape[0], -1), (0, self.padding)).reshape(
            *scale.shape[:-1], -1)
        # The zero chromosome reproduces PyTorch ties-to-even RTN exactly.
        codes = (blocks/scale + .1*int(gene[1])).round().clamp(-8, 7)
        self.cached.copy_((codes*scale).reshape(self.shape[0], -1)[:, :self.width].reshape(self.shape))

    def forward(self, original):
        return self.cached if self.enabled else original


def split_records(dataset, count, seed):
    """Prompt-disjoint FIT/SELECT subsets inside TRAIN; spread across timesteps."""
    if not dataset or any(len(row) != 4 for row in dataset):
        raise ValueError('Evolution requires trajectory calibration')
    groups = {}
    for i, row in enumerate(dataset):
        groups.setdefault(row[2], []).append(i)
    if len(groups) < 4:
        raise ValueError('Evolution requires at least four distinct TRAIN prompts')
    rng = np.random.default_rng(seed)
    prompts = sorted(groups)
    rng.shuffle(prompts)
    split = len(prompts)//2
    result = []
    for subset in (prompts[:split], prompts[split:]):
        # Each chosen prompt contributes a different timestep stratum, cycling.
        indices = []
        for j in range(count):
            rows = sorted(groups[subset[j % len(subset)]], key=lambda i: dataset[i][3])
            indices.append(rows[(j//len(subset)+j) % len(rows)])
        result.append(indices)
    return result


@torch.no_grad()
def train_evolution(pipe, scheduler, dataset, names, method, args, output):
    from wmq_sleepermark import encode_text, snapshot
    from wmq_blind import save_csv, save_json
    from wmq_sleeper_calibration import augmented_prompts
    from wmq_sleeper_equivariance import guided_prediction
    started = time.perf_counter()
    device = next(pipe.unet.parameters()).device
    empty = encode_text(pipe, ['']).detach()
    pipe.unet.eval()
    if args.quant_group_size < 1:
        raise ValueError('Evolution needs grouped W4 (--quant-group-size > 0)')
    fit_indices, select_indices = split_records(dataset, args.evolution_records, args.seed)
    groups = sorted(set(n.split('.transformer_blocks.')[0] if '.transformer_blocks.' in n
                        else n.rsplit('.', 2)[0] for n in names))
    group_index = {name: i for i, name in enumerate(groups)}
    modules, rows, calls = [], [], [0]
    def count_forward(module, inputs):
        calls[0] += 1
    counter = pipe.unet.register_forward_pre_hook(count_forward)
    def enable(value):
        for _, _, adapter, _ in modules:
            adapter.enabled = value
    def apply(gene):
        for module, leaf, adapter, group in modules:
            adapter.set_gene(getattr(module.parametrizations, leaf).original, gene[2*group:2*group+2])
        enable(True)
    try:
        for name in names:
            path, leaf = name.rsplit('.', 1)
            module = pipe.unet.get_submodule(path)
            adapter = GeneticWeight(getattr(module, leaf), args.quant_group_size, args.weight_init)
            parametrize.register_parametrization(module, leaf, adapter)
            group = name.split('.transformer_blocks.')[0] if '.transformer_blocks.' in name else name.rsplit('.', 2)[0]
            modules.append((module, leaf, adapter, group_index[group]))
        rng = np.random.default_rng(args.seed+7919)
        def bank(indices):
            result = []
            for i in indices:
                z, context, prompt, timestep = dataset[i]
                z, context = z.to(device), context.to(device)
                t = torch.tensor([timestep], device=device, dtype=torch.long)
                u = pipe.unet(z, t, encoder_hidden_states=empty).sample
                c = pipe.unet(z, t, encoder_hidden_states=context).sample
                # Random ordinary prefixes, independent of owner secret.
                augmented = encode_text(pipe, augmented_prompts([prompt], rng)).detach()
                # Proxy calibration restricted to late time; early records still
                # contribute ordinary quality over the trajectory.
                target = None
                if timestep <= args.late_fraction*scheduler.config.num_train_timesteps:
                    target, _ = guided_spatial_target(pipe.unet, z, t, augmented, empty,
                        (args.spatial_shift, -args.spatial_shift), scheduler, args.max_spatial_correction)
                result.append((z, t, context, u.detach(), (u+7.5*(c-u)).detach(), augmented, target))
            if not any(row[-1] is not None for row in result):
                raise ValueError('Evolution calibration bank has no late timestep; increase records/trajectory points')
            return result
        enable(False)
        fit_bank, select_bank = bank(fit_indices), bank(select_indices)
        def score(data):
            ordinary, proxy = [], []
            for z, t, context, u, cfg, augmented, target in data:
                pred_u = pipe.unet(z, t, encoder_hidden_states=empty).sample
                pred_c = pipe.unet(z, t, encoder_hidden_states=context).sample
                ordinary.append(F.mse_loss(pred_u+7.5*(pred_c-pred_u), cfg)/7.5**2+F.mse_loss(pred_u, u))
                if target is not None:
                    guided = guided_prediction(pipe.unet, z, t, augmented, empty)
                    proxy.append(F.mse_loss(predicted_x0(guided, z, t, scheduler), target))
            return torch.stack([torch.stack(ordinary).mean(), torch.stack(proxy).mean()]).cpu().numpy()
        def fitness(gene):
            apply(gene)
            value = score(fit_bank)
            rows.append({'candidate': len(rows), 'ordinary_loss': float(value[0]), 'proxy_loss': float(value[1]),
                         'elapsed_seconds': time.perf_counter()-started,
                         'logical_unet_forwards': calls[0],
                         'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.})
            if len(rows) % args.evolution_population == 0:
                save_csv(output/(method+'_training.csv'), rows)
                print({'method': method, 'evaluated': len(rows), **rows[-1]}, flush=True)
            return value
        records, pareto = search(fitness, 2*len(groups), args.evolution_population,
                                args.evolution_generations, args.seed, random=method == 'random_quantizer_w4')
        # Predeclared holdout budget: anchor plus P-1 diverse fit-Pareto/ranked
        # candidates; SELECT is disjoint from FIT and from owner TEST prompts.
        shortlist = [0]+[i for i in ranked([r['fitness'] for r in records]) if i != 0][:args.evolution_population-1]
        validation = []
        for i in shortlist:
            apply(records[i]['gene'])
            ordinary, proxy = map(float, score(select_bank))
            validation.append({'candidate': i, 'ordinary_loss': ordinary, 'proxy_loss': proxy})
        limit = max(validation[0]['ordinary_loss']*args.evolution_quality_ratio, 1e-12)
        feasible = [v for v in validation if v['ordinary_loss'] <= limit]
        winner = min(feasible, key=lambda v: (v['proxy_loss'], v['ordinary_loss'], v['candidate']))
        apply(records[winner['candidate']]['gene'])
        save_csv(output/(method+'_validation.csv'), validation)
        save_json(output/(method+'_search.json'), {'algorithm': 'random' if method.startswith('random') else 'NSGA-II-style integer GA',
            'groups': groups, 'gene_meaning': ['log_scale/.04', 'rounding_bias/.1'], 'bounds': [-3, 3],
            'records': records, 'fit_pareto_indices': pareto, 'selected': winner,
            'ordinary_loss_limit': limit, 'quality_ratio_to_rtn': args.evolution_quality_ratio,
            'fit_prompt_names': sorted(set(dataset[i][2] for i in fit_indices)),
            'select_prompt_names': sorted(set(dataset[i][2] for i in select_indices)),
            'fit_record_indices': fit_indices, 'select_record_indices': select_indices,
            'test_used_for_selection': False, 'owner_used_for_fitness': False,
            'proxy_warning': 'Spatial target is not an identified watermark. Quality constraint is calibration noise MSE, not image SSIM.',
            'elapsed_seconds': time.perf_counter()-started, 'logical_unet_forwards': calls[0]})
        save_csv(output/(method+'_training.csv'), rows)
        return {method: snapshot(pipe.unet, names)}
    finally:
        counter.remove()
        for module, leaf, _, _ in reversed(modules):
            parametrize.remove_parametrizations(module, leaf, leave_parametrized=False)
        pipe.unet.eval()
