"""Small, deterministic integer-genome NSGA-II style search (two losses, minimize).

No torch, model, or owner evaluation dependencies. Random control gets the same
initial population, bounds and number of fitness calls as evolution.
"""
import numpy as np


def fronts(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Fitness must be a finite matrix')
    remaining = list(range(len(values)))
    result = []
    while remaining:
        front = [i for i in remaining if not any(
            np.all(values[j] <= values[i]) and np.any(values[j] < values[i])
            for j in remaining if j != i)]
        result.append(front)
        members = set(front)
        remaining = [i for i in remaining if i not in members]
    return result


def crowding(values, indices):
    distance = {i: 0. for i in indices}
    for objective in range(len(values[0])):
        order = sorted(indices, key=lambda i: values[i][objective])
        span = values[order[-1]][objective] - values[order[0]][objective]
        if span <= 0:
            continue
        distance[order[0]] = distance[order[-1]] = float('inf')
        for left, current, right in zip(order, order[1:], order[2:]):
            distance[current] += (values[right][objective]-values[left][objective])/span
    return distance


def ranked(values):
    result = []
    for front in fronts(values):
        spread = crowding(values, front)
        result.extend(sorted(front, key=lambda i: (-spread[i], i)))
    return result


def search(evaluate, dimensions, population=12, generations=12, seed=0, random=False):
    """Integer genes in [-3,3]. Zero anchor is measured, not assumed optimal.

    Elitist parent+offspring selection, Pareto rank/crowding tournaments,
    uniform crossover and discrete mutation. Exactly P*(G+1) evaluations.
    """
    if dimensions < 1 or population < 4 or generations < 1:
        raise ValueError('Need dimensions>0, population>=4, generations>=1')
    rng = np.random.default_rng(seed)
    records = []

    def measure(gene, generation):
        score = np.asarray(evaluate(gene.copy()), dtype=float)
        if score.shape != (2,) or not np.isfinite(score).all():
            raise ValueError('Nonfinite or malformed candidate fitness')
        records.append({'gene': gene.tolist(), 'fitness': score.tolist(), 'generation': generation})
        return len(records)-1

    initial = rng.integers(-3, 4, size=(population, dimensions))
    initial[0] = 0
    live = [measure(gene, 0) for gene in initial]
    for generation in range(1, generations+1):
        order = ranked([records[i]['fitness'] for i in live])
        priority = {live[index]: rank for rank, index in enumerate(order)}
        def tournament():
            a, b = rng.choice(live, 2, replace=False)
            return np.asarray(records[min((a, b), key=priority.get)]['gene'])
        children = []
        for _ in range(population):
            if random:
                gene = rng.integers(-3, 4, dimensions)
            else:
                a, b = tournament(), tournament()
                gene = np.where(rng.random(dimensions) < .5, a, b)
                mutation = rng.random(dimensions) < max(1/dimensions, .1)
                mutation[rng.integers(dimensions)] = True
                gene = np.clip(gene + mutation*rng.choice([-2, -1, 1, 2], dimensions), -3, 3)
            children.append(measure(gene, generation))
        pool = live + children
        live = [pool[i] for i in ranked([records[j]['fitness'] for j in pool])[:population]]
    return records, [i for i in fronts([r['fitness'] for r in records])[0]]
