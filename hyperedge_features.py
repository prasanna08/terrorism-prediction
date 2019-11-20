from collections import defaultdict
import scipy.sparse as sp
from functools import reduce
import numpy as np
import networkx as nx

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

def arithmatic_mean(vals):
    return sum(vals) / float(len(vals))

def geomatric_mean(vals):
    return np.pow(np.prod(vals), 1.0/len(vals))

def harmonic_mean(vals):
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
    mean(
        [1.0 / np.log10(len(nodes_to_neighbors[i]))
         for i in edge for j in edge
         for node in nodes_to_neighbors[i].intersection(nodes_to_neighbors[j])
         if i < j],
        mean_type='harmonic')

def hyper_clustering_coefficient(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors):
    neighbor_hyper_edges = [
        edgeidx for nedge, edgeidx in enumerate(hyperedges)
        if len(nedge.intersection(edge)) > 0]
    return mean(
        [1 if j in dual_nodes_to_neighbors[i] else 0
         for i in neighbor_hyper_edges
         for j in neighbor_hyper_edges if i < j])

def hyper_jaccard_coefficient(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [len(nodes_to_neighbors[i].intersection(nodes_to_neighbors[j])) / float(
            len(nodes_to_neighbors[i].union(nodes_to_neighbors[j])))
         for i in edge for j in edge if i < j])

def hyper_edge_density(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean(
        [1 if i in nodes_to_neighbors[j] else 0
         for i in edge for j in edge if i < j])

def hyper_preferential_attachment(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean([len(nodes_to_neighbors[i]) for i in edge], mean_type='geomatric')

def hyper_shortest_distance(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, *vargs):
    return mean(
        [shortest_path_lengths[i][j] for i in edge for j in edge if i > j])

def hyper_avg_degree(edge, hyperedges, nodes_to_neighbors, *vargs):
    return mean([len(nodes_to_neighbors[node]) for node in edge])

def hyper_salton_index(edge, hyperedges, nodes_to_neighbors, *vargs):
    common_nodes = reduce(
        lambda x, y: x.intersection(y),
        [nodes_to_neighbors[node] for node in edge])
    gmean = mean(
        [len(nodes_to_neighbors[node]) for node in egde], mean_type='geomatric')
    return len(common_nodes) / gmean

def hyper_sorenson_index(edge, hyperedges, nodes_to_neighbors, *vargs):
    common_nodes = reduce(
        lambda x, y: x.intersection(y),
        [nodes_to_neighbors[node] for node in edge])
    amean = mean([len(nodes_to_neighbors[node]) for node in egde])
    return len(common_nodes) / amean

def hyper_katz(
        edge, hyperedges, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, katz, *vargs):
    return mean(
        [katz[node1, node2] for node1 in edge for node2 in edge
         if node1 < node2])

def preprocess(H):
    H = sp.csr_matrix(H)
    A = np.dot(H, H.T)
    D = np.dot(H.T, H)
    A[A > 0] = 1
    D[D > 0] = 1
    katz = np.linalg.pinv(np.eye(A.shape[0]) - 0.05 * A) - np.eye(A.shape[0])
    G = nx.from_numpy_matrix(A)
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    nodes_to_neighbors = defaultdict(set)
    for i, j in zip(*A.nonzero()):
        nodes_to_neighbors[i].add(j)
        nodes_to_neighbors[j].add(i)
    dual_nodes_to_neighbors = defaultdict(set)
    for i, j in zip(*D.nonzero()):
        dual_nodes_to_neighbors[i].add(j)
        dual_nodes_to_neighbors[j].add(i)
    return (
        H, A, D, nodes_to_neighbors, dual_nodes_to_neighbors,
        shortest_path_lengths, katz)

def hyperedge_featurizer(hyperedges, H, features):
    H, A, D, nodes_to_neighbors, dual_nodes_to_neighbors, \
        shortest_path_lengths, katz = preprocess(H)
    hyperedge_features = []
    
    if features is None:
        features = FEATURE_MAP.keys()
    
    for edgeidx, edge in enumerate(hyperedges):
        edge_features = {}
        for feat in features:
            val = FEATURE_MAP[feat](
                edge, edgeidx, nodes_to_neighbors,
                dual_nodes_to_neighbors, shortest_path_lengths, katz)
            edge_features[feat] = val
        edge_features['hyperedge'] = edge
        hyperedge_features.append(edge_features)
    feature_matrix = []
    for edge_features in hyperedge_features:
        feature_matrix.append(edge_features[feat] for feat in features)
    return hyperedge_features, np.array(feature_matrix)
