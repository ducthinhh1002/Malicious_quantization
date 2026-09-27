"""Bounded candidate proposals from FIT-only feedback; no generated code execution."""
import json
import time
from pathlib import Path

import numpy as np
from wmq_evolution import ranked

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def fit_context(groups, archive, generation, count):
    # Whitelist fields, never serialize a model report or args namespace here.
    indices = list(dict.fromkeys([0]+ranked([r['fitness'] for r in archive])[:10]+
                                 list(range(max(0, len(archive)-4), len(archive)))))
    return {'generation': generation, 'groups': list(groups), 'proposal_count': count,
            'gene_order': 'for each group: integer log_scale/.04, integer rounding_bias/.1',
            'bounds': [-3, 3], 'objective_order': ['ordinary_noise_loss', 'residual_proxy_loss'],
            'objectives': 'Both minimized. Residual proxy is not a measured watermark score.',
            'fit_archive': [{'id': i, 'gene': list(archive[i]['gene']),
                             'fitness': list(archive[i]['fitness'])} for i in indices]}


def parse_proposals(response, dimensions, count):
    text = response.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        if text.startswith('json\n'):
            text = text[5:].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'genes'} or not isinstance(value['genes'], list):
        raise ValueError('Expected JSON object with only genes array')
    if not 1 <= len(value['genes']) <= count:
        raise ValueError('Invalid proposal count')
    for gene in value['genes']:
        if (not isinstance(gene, list) or len(gene) != dimensions or
                any(type(g) is not int or not -3 <= g <= 3 for g in gene)):
            raise ValueError('Every gene must be an integer in [-3,3] of the declared dimension')
    return value['genes']


class LocalEliteProposer:
    """Cheap locality control: one-coordinate neighbors of FIT Pareto elites."""
    label = 'local_elite_proposal'

    def __init__(self, groups, count=3, seed=0):
        self.groups, self.count = groups, count
        self.rng = np.random.default_rng(seed+65537)
        self.trace = []

    def __call__(self, generation, archive):
        order = ranked([r['fitness'] for r in archive])[:4]
        seen = {tuple(r['gene']) for r in archive}
        result = []
        for _ in range(self.count*32):
            gene = list(archive[int(self.rng.choice(order))]['gene'])
            coordinate = int(self.rng.integers(len(gene)))
            gene[coordinate] = int(np.clip(gene[coordinate]+self.rng.choice([-1, 1]), -3, 3))
            if tuple(gene) not in seen:
                result.append(gene)
                seen.add(tuple(gene))
            if len(result) == self.count:
                break
        self.trace.append({'generation': generation, 'proposed': len(result)})
        return result

    def close(self):
        pass


class LocalLLMProposer:
    label = 'llm_proposal'

    def __init__(self, groups, output, count=3, max_tokens=768, device='cuda'):
        self.groups, self.output, self.count = groups, Path(output), count
        self.max_tokens, self.device = max_tokens, device
        self.trace, self.model, self.tokenizer = [], None, None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        dtype = torch.bfloat16 if self.device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=dtype, trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, generation, archive):
        import torch
        started = time.perf_counter()
        if self.model is None:
            self._load()  # Loading failure aborts this branch; never masquerade as LLM success.
        context = fit_context(self.groups, archive, generation, self.count)
        messages = [
            {'role': 'system', 'content': 'You propose bounded integer configurations for W4 quantization. '
             'Use the supplied FIT measurements only. Both losses are minimized. '
             'Preserve ordinary quality while reducing the residual proxy. '
             'Prefer small coordinated changes to good configurations and avoid repeated genomes. '
             'Return only valid JSON: {"genes":[[integer,...],...]}. No explanation or code. '
             f'Each genome has exactly {2*len(self.groups)} integers. Return at most {self.count} genomes.'},
            {'role': 'user', 'content': json.dumps(context, allow_nan=False)}]
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                     return_tensors='pt').to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(encoded, attention_mask=torch.ones_like(encoded),
                max_new_tokens=self.max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True)
        error, genes = None, []
        try:
            genes = parse_proposals(response, 2*len(self.groups), self.count)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        seen = {tuple(r['gene']) for r in archive}
        fresh = []
        for gene in genes:
            if tuple(gene) not in seen:
                fresh.append(gene)
                seen.add(tuple(gene))
        entry = {'generation': generation, 'model': MODEL, 'revision': REVISION,
                 'messages': messages, 'response': response, 'parse_error': error,
                 'input_tokens': encoded.shape[1], 'output_tokens': generated.shape[1]-encoded.shape[1],
                 'proposed': len(genes), 'accepted_unique': len(fresh),
                 'seconds': time.perf_counter()-started, 'owner_feedback': False}
        self.trace.append(entry)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print({'llm_generation': generation, 'accepted_unique': len(fresh),
               'parse_error': error, 'seconds': entry['seconds']}, flush=True)
        return fresh

    def close(self):
        import gc
        import torch
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
