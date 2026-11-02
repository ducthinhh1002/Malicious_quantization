"""Decode a key from mean normalized carrier coordinates, without a TPR threshold."""
import numpy as np


def aggregate_metrics(coordinates, signs, sizes=(1,8,32,64,256), repeats=100, seed=20261029):
    q=np.asarray(coordinates,dtype=np.float64);key=np.asarray(signs)
    if q.ndim!=2 or q.shape[1]!=len(key) or not np.isfinite(q).all():
        raise ValueError('Invalid carrier coordinates')
    if not np.isin(key,[-1,1]).all():raise ValueError('Invalid key')
    rows=[];n=len(q)
    for size in sizes:
        groups=n//size
        if not groups:continue
        rng=np.random.default_rng(seed+size);ba=[];fk=[]
        for _ in range(1 if size==1 else repeats):
            ids=rng.permutation(n)[:groups*size].reshape(groups,size)
            means=q[ids].mean(1)
            decoded=np.where(means>=0,1,-1)
            correct=decoded==key
            ba.append(float(correct.mean()));fk.append(float(correct.all(1).mean()))
        rows.append({'images_per_group':size,'bit_accuracy':float(np.mean(ba)),
            'full_key_recovery_rate':float(np.mean(fk)),
            'groups_per_partition':groups,'partitions':len(ba),
            'discarded_per_partition':n-groups*size,
            'partition_sd_bit_accuracy':float(np.std(ba)),
            'partition_sd_full_key_recovery':float(np.std(fk))})
    return {'n_images':n,'num_bits':len(key),'rows':rows,'grouping_seed':seed,
        'decode':'mean normalized carrier coordinates within group; >=0 -> +1 else -1',
        'grouping':'100 fixed random non-overlapping partitions; no repeated image within group; size1 once',
        'full_key':'all payload bits correct in a group',
        'uncertainty':'partition SD measures grouping variability, not independent-sample confidence intervals',
        'all_images_aggregate_bit_accuracy':float((np.where(q.mean(0)>=0,1,-1)==key).mean())}
