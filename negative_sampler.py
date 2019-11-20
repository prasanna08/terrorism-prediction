from collections import Counter
import numpy as np

def get_size_distribution(hyperedges):
    size_dist = dict(Counter([len(edge) for edge in hyperedges]).items())
    if 0 in size_dist:
        del size_dist[0]
    if 1 in size_dist:
        del size_dist[1]
    total = sum(size_dist.values())
    for k in size_dist:
        size_dist[k] = size_dist[k] / float(total)
    return size_dist

def sample_negative_hyperedges(hyperedges, total_nodes, count):
    size_dist = get_size_distribution(hyperedges)
    hyperedge_length = [k for k, _ in size_dist.items()]
    pvals = [v for _, v in size_dist.items()]
    sampled_sizes = np.random.choice(hyperedge_length, pvals=pvals, size=count)
    negative_hyperedges = set()
    hyperedges = set(hyperedges)
    while len(negative_hyperedges) < count:
        sampled_size = sampled_sizes[len(negative_hyperedges)]
        nodes = frozenset(np.random.choice(total_nodes, size=sampled_size, replace=False))
        if nodes not in hyperedges and nodes not in negative_hyperedges:
            negative_hyperedges.add(nodes)
    return list(negative_hyperedges)
