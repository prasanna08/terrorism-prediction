from collections import defaultdict
import scipy.sparse as sp
from functools import reduce
import numpy as np
import networkx as nx
from tqdm import tqdm

def arithmatic_mean(vals):
    if len(vals) == 0:
        return 0
    return sum(vals) / float(len(vals))

def geomatric_mean(vals):
    if len(vals) == 0 or np.prod(vals) == 0:
        return 0
    return np.prod(np.power(vals, 1.0/len(vals)))

def harmonic_mean(vals):
    if len(vals) == 0:
        return 0
    return 1.0 / mean([1.0/v for v in vals])

def mean(vals, mean_type='arithmatic'):
    if mean_type == 'arithmatic':
        return arithmatic_mean(vals)
    elif mean_type == 'geomatric':
        return geomatric_mean(vals)
    else:
        return harmonic_mean(vals)

def hyper_common_neighbors(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [len(nodes_to_neighbors[i].intersection(nodes_to_neighbors[j]))
         for i in edge for j in edge if i < j])

def hyper_adamic_adar(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [1.0 / np.log10(len(nodes_to_neighbors[node]))
         for i in edge for j in edge
         for node in nodes_to_neighbors[i].intersection(nodes_to_neighbors[j])
         if i < j],
        mean_type='harmonic')

def hyper_clustering_coefficient(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors, *vargs):
    neighbor_hyper_edges = [
        edgeidx for edgeidx, nedge in enumerate(hyperedges)
        if len(nedge.intersection(edge)) > 0]
    return mean(
        [1 if j in dual_nodes_to_neighbors[i] else 0
         for i in neighbor_hyper_edges
         for j in neighbor_hyper_edges if i < j])

def hyper_jaccard_coefficient(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [(len(nodes_to_neighbors[i].intersection(nodes_to_neighbors[j])) / float(
            len(nodes_to_neighbors[i].union(nodes_to_neighbors[j]))))
            if len(nodes_to_neighbors[i].union(nodes_to_neighbors[j])) > 0 else 0
         for i in edge for j in edge if i < j])

def hyper_edge_density(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [1 if i in nodes_to_neighbors[j] else 0
         for i in edge for j in edge if i < j])

def hyper_preferential_attachment(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean([
        len(nodes_to_neighbors[i]) for i in edge], mean_type='geomatric')

def hyper_shortest_distance(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, *vargs):
    return mean([
        shortest_path_lengths[i][j] if j in shortest_path_lengths[i]
        else (len(nodes_to_neighbors) + 1) for i in edge for j in edge if i > j])

def hyper_avg_degree(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean([len(nodes_to_neighbors[node]) for node in edge])

def hyper_salton_index(edge, hyperedges, nodes_to_neighbors, *vargs):
    common_nodes = reduce(
        lambda x, y: x.intersection(y),
        [nodes_to_neighbors[node] for node in edge])
    gmean = mean(
        [len(nodes_to_neighbors[node]) for node in edge], mean_type='geomatric')
    return len(common_nodes) / gmean if gmean > 0 else 0

def hyper_sorenson_index(edge, hyperedges, nodes_to_neighbors, *vargs):
    common_nodes = reduce(
        lambda x, y: x.intersection(y),
        [nodes_to_neighbors[node] for node in edge])
    amean = mean([len(nodes_to_neighbors[node]) for node in edge])
    return len(common_nodes) / amean if amean > 0 else 0

def hyper_katz(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, katz, *vargs):
    return mean(
        [katz[node1, node2] for node1 in edge for node2 in edge
         if node1 < node2])

FEATURE_MAP = {
    'HCN': hyper_common_neighbors,
    'HAA': hyper_adamic_adar,
    'HCC': hyper_clustering_coefficient,
    'HJC': hyper_jaccard_coefficient,
    'HED': hyper_edge_density,
    'HPA': hyper_preferential_attachment,
    'HSoI': hyper_sorenson_index,
    'HSaI': hyper_salton_index,
    'HSD': hyper_shortest_distance,
    'Hkatz': hyper_katz,
    'HAD': hyper_avg_degree
}

def preprocess(H):
    print('Preprocessing Hypergraph information')
    H = sp.csr_matrix(H)
    A = H * H.T
    D = H.T * H
    A[A > 0] = 1
    D[D > 0] = 1
    A.setdiag([0] * A.shape[0])
    D.setdiag([0] * D.shape[0])

    print('Calculating Katz Matrix')
    katz = np.linalg.pinv(np.eye(A.shape[0]) - 0.05 * A) - np.eye(A.shape[0])
    G = nx.from_numpy_matrix(A.toarray())

    print('Finding all pairs shortest paths')
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    nodes_to_neighbors = defaultdict(set)
    for i, j in zip(*A.nonzero()):
        nodes_to_neighbors[i].add(j)
        nodes_to_neighbors[j].add(i)
    dual_nodes_to_neighbors = defaultdict(set)
    for i, j in zip(*D.nonzero()):
        dual_nodes_to_neighbors[i].add(j)
        dual_nodes_to_neighbors[j].add(i)
    
    print('Preprocessing stage completed')
    return (
        H, A, D, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, katz)

def hyperedge_featurizer(hyperedges, H, features):
    H, A, D, nodes_to_neighbors, dual_nodes_to_neighbors, \
        shortest_path_lengths, katz = preprocess(H)
    hyperedge_features = []
    
    if features is None:
        features = FEATURE_MAP.keys()
    
    print('Calculating Hyperedge features')
    for edgeidx, edge in enumerate(tqdm(hyperedges)):
        edge_features = {}
        for feat in features:
            val = FEATURE_MAP[feat](
                edge, hyperedges, nodes_to_neighbors,
                dual_nodes_to_neighbors, shortest_path_lengths, katz)
            edge_features[feat] = val
        edge_features['hyperedge'] = edge
        hyperedge_features.append(edge_features)
    feature_matrix = []
    for edge_features in hyperedge_features:
        feature_matrix.append([edge_features[feat] for feat in features])
    return hyperedge_features, np.array(feature_matrix)
