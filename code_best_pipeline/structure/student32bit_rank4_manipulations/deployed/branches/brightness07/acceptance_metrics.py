"""BA, exact FK32bits, and acceptance >=24/32; never conflate the last two."""
import math
import numpy as np

def aggregate_metrics(coordinates, signs, sizes=(1,8,32,64,256), repeats=100, seed=20261029):
    q = np.asarray(coordinates, dtype=np.float64)
    key = np.asarray(signs)
    if q.ndim != 2 or q.shape[1] != 32 or key.shape != (32,):
        raise ValueError('Expected 32-bit coordinates/key')
    if not np.isfinite(q).all() or not np.isin(key, [-1,1]).all():
        raise ValueError('Invalid coordinates/key')
    rows=[]
    for size in sizes:
        ng=len(q)//size
        if not ng: continue
        rng=np.random.default_rng(seed+size)
        counts=[]
        for _ in range(1 if size == 1 else repeats):
            ids=rng.permutation(len(q))[:ng*size].reshape(ng,size)
            correct=(np.where(q[ids].mean(1)>=0,1,-1)==key).sum(1)
            counts.extend(correct.tolist())
        counts=np.asarray(counts)
        rows.append({'images_per_group':size,'bit_accuracy':float(counts.mean()/32),
                     'exact_full_key_rate':float((counts==32).mean()),
                     'key_acceptance_ge24_rate':float((counts>=24).mean()),
                     'accepted':int((counts>=24).sum()),'groups_evaluated':len(counts),
                     'discarded_per_partition':len(q)-ng*size})
    return {'n_images':len(q),'num_bits':32,'rows':rows,
            'theoretical_fpr_ge24':sum(math.comb(32,k) for k in range(24,33))/2**32,
            'fpr_scope':'Binomial null with independent fair bit matches; not empirical FPR',
            'grouping_seed':seed,'partitions':repeats,
            'note':'Groups overlap across partitions; not independent trials',
            'per_bit_accuracy':(np.where(q>=0,1,-1)==key).mean(0).tolist(),
            'per_bit_signed_mean':(q*key).mean(0).tolist()}
