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


def ranked(values, behaviors=None):
    result = []
    features = None
    if behaviors is not None:
        features = np.asarray(behaviors, dtype=float)
        if features.ndim != 2 or len(features) != len(values) or not np.isfinite(features).all():
            raise ValueError('Finite behavior vectors required for every candidate')
        features = features/np.maximum(np.std(features, axis=0), 1e-8)
    for front in fronts(values):
        spread = crowding(values, front)
        order = sorted(front, key=lambda i: (-spread[i], i))
        if features is not None and len(front) > 2:
            # Preserve objective extremes, then retain distinct observed model
            # responses within this Pareto rank. Never use owner/test features.
            chosen = list(dict.fromkeys(min(front, key=lambda i: values[i][j]) for j in range(len(values[0]))))
            pending = [i for i in order if i not in chosen]
            while pending:
                index = max(pending, key=lambda i: (min(float(np.square(features[i]-features[j]).mean())
                                                       for j in chosen), spread[i], -i))
                chosen.append(index)
                pending.remove(index)
            order = chosen
        result.extend(order)
    return result


def operator_probabilities(credits):
    credits = np.asarray(credits, dtype=float)
    merit = np.maximum(credits, 0)+.05
    return .2/len(merit)+.8*merit/merit.sum()


def paired_crossover(a, b, rng, group_size=2):
    if len(a) % group_size:
        raise ValueError('Genome must contain complete crossover groups')
    return np.where(np.repeat(rng.random(len(a)//group_size) < .5, group_size), a, b)


def search(evaluate, dimensions, population=12, generations=12, seed=0, random=False,
           structured=False, behavior_selection=False, adaptive=False, paired=True):
    """Integer genes in [-3,3]. Zero anchor is measured, not assumed optimal.

    Elitist parent+offspring selection, Pareto rank/crowding tournaments,
    uniform crossover and discrete mutation. Exactly P*(G+1) distinct genomes
    evaluated; reject budgets larger than the finite search space.
    """
    if dimensions < 1 or population < 4 or generations < 1:
        raise ValueError('Need dimensions>0, population>=4, generations>=1')
    if population*(generations+1) > 7**dimensions:
        raise ValueError('Unique evaluation budget exceeds the integer search space')
    if structured and paired and dimensions % 2:
        raise ValueError('Paired search needs an even number of genes')
    rng = np.random.default_rng(seed)
    records = []
    seen = set()
    credits = np.zeros(4)
    operator_names = ('single_mutation', 'group_mutation', 'group_crossover', 'immigrant')

    def unseen(gene):
        # Duplicates arise from crossover, saturated mutation and elitism.
        # Try random immigrants without spending a model forward. Near space
        # exhaustion, deterministic enumeration guarantees termination.
        for _ in range(64):
            if tuple(gene) not in seen:
                return gene
            gene = rng.integers(-3, 4, dimensions)
        for code in range(len(seen)+1):
            gene = np.empty(dimensions, dtype=int)
            for j in range(dimensions):
                gene[j] = code % 7-3
                code //= 7
            if tuple(gene) not in seen:
                return gene
        raise RuntimeError('No unseen chromosome despite valid budget')

    def measure(gene, generation, operator='initial', probabilities=None):
        duplicate = tuple(gene) in seen
        gene = unseen(gene)
        seen.add(tuple(gene))
        evaluated = evaluate(gene.copy())
        behavior = evaluated.get('behavior') if isinstance(evaluated, dict) else None
        score = np.asarray(evaluated['fitness'] if isinstance(evaluated, dict) else evaluated, dtype=float)
        if score.shape != (2,) or not np.isfinite(score).all():
            raise ValueError('Nonfinite or malformed candidate fitness')
        record = {'gene': gene.tolist(), 'fitness': score.tolist(), 'generation': generation,
                  'operator': 'duplicate_restart' if duplicate else operator}
        if behavior_selection:
            vector = np.asarray(behavior, dtype=float)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError('Behavior selection requires finite nonempty descriptors')
            record['behavior'] = vector.tolist()
        if probabilities is not None:
            record['operator_probabilities'] = probabilities.tolist()
        records.append(record)
        return len(records)-1

    def population_order(indices):
        return ranked([records[i]['fitness'] for i in indices],
                      [records[i]['behavior'] for i in indices] if behavior_selection else None)

    initial = rng.integers(-3, 4, size=(population, dimensions))
    initial[0] = 0
    live = [measure(gene, 0) for gene in initial]
    for generation in range(1, generations+1):
        order = population_order(live)
        priority = {live[index]: rank for rank, index in enumerate(order)}
        def tournament_index():
            a, b = rng.choice(live, 2, replace=False)
            return min((a, b), key=priority.get)
        children = []
        for _ in range(population):
            op = None
            probabilities = operator_probabilities(credits) if adaptive else np.full(4, .25)
            parent = None
            if random:
                gene = rng.integers(-3, 4, dimensions)
            elif structured:
                op = int(rng.choice(4, p=probabilities))
                parent = tournament_index()
                gene = np.asarray(records[parent]['gene']).copy()
                group_size = 2 if paired else 1
                if op == 0:
                    gene[rng.integers(dimensions)] += rng.choice([-2, -1, 1, 2])
                elif op == 1:
                    start = int(rng.integers(dimensions//group_size))*group_size
                    gene[start:start+group_size] += rng.choice([-2, -1, 1, 2], group_size)
                elif op == 2:
                    other = np.asarray(records[tournament_index()]['gene'])
                    gene = paired_crossover(gene, other, rng, group_size)
                else:
                    gene = rng.integers(-3, 4, dimensions)
                gene = np.clip(gene, -3, 3)
            else:
                a, b = (np.asarray(records[tournament_index()]['gene']) for _ in range(2))
                gene = np.where(rng.random(dimensions) < .5, a, b)
                mutation = rng.random(dimensions) < max(1/dimensions, .1)
                mutation[rng.integers(dimensions)] = True
                gene = np.clip(gene + mutation*rng.choice([-2, -1, 1, 2], dimensions), -3, 3)
            child = measure(gene, generation, operator_names[op] if op is not None else 'random' if random else 'uniform',
                            probabilities if structured else None)
            children.append(child)
            if adaptive and parent is not None and records[child]['operator'] != 'duplicate_restart':
                before, after = np.asarray(records[parent]['fitness']), np.asarray(records[child]['fitness'])
                # A success improves at least one objective without being
                # dominated by its parent; final survival still uses all peers.
                reward = float(np.any(after < before) and not (np.all(before <= after) and np.any(before < after)))
                credits[op] = .9*credits[op]+.1*reward
                records[child]['operator_reward'] = reward
        pool = live + children
        live = [pool[i] for i in population_order(pool)[:population]]
    return records, [i for i in fronts([r['fitness'] for r in records])[0]]
