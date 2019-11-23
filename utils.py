import scipy.sparse as sp
import numpy as np

def hyperedges_to_incidence_matrix(hyperedges, total_nodes):
    '''Tansfrom a list of hyperedges into incidence matrix hypergraph.

    Args:
    hyperedges: List(frozenset). A list of hyperedges where each hyperedge is
        frozenset of nodes it contains.
    total_nodes: Int. Total number of nodes in hyperedge.

    Returns:
    H: scipy.sparse.csr_matrix. Sparse matrix representation of hyperedges.
    '''
    edgevectors = []
    for hedge in hyperedges:
        evec = [0 for _ in range(total_nodes)]
        for i in hedge:
            evec[i] = 1
        edgevectors.append(evec)
    H = sp.csr_matrix(np.array(edgevectors).T)
    return H

def incidence_matrix_to_hyperedges(H):
    '''Transfrom incidence matrix representation of hypergraph H into list of
    hyperedges.

    Args:
    H: scipy.sparse.csr_matrix. Sparse matrix representation of hypergraph H.

    Returns:
    hyperedges: List(fronzenset). List of hyperedges where each hyperedge is
        frozenset of nodes it contains.
    '''
    H = H.toarray()
    hyperedges = []
    for row in H.T:
        hyperedges.append(frozenset(np.where(row > 0)[0].tolist()))
    return hyperedges

def network_deconvolution(A, threshold=0.35):
    '''Deconvolution of a network represented by adjacency matrix A.

    Args:
    A: np.array. Adjacency matrix of the network.
    threshold: float. All edges with weeight below threshold are removed from 
        deconvoluted matrix and the weight for rest is set to 1.

    Returns:
    D: np.array. Deconvoluted adjacency matrix of network.
    '''
    A[A > 0] = 1
    Ap = np.array(np.linalg.pinv(np.identity(A.shape[0]) + A))
    A_deconv = np.dot(A, Ap)
    A_deconv[A_deconv > threshold] = 1
    A_deconv[A_deconv <= threshold] = 0
    return A_deconv

def split_hyperedges_according_to_time_for_ty2001_06_dataset(
        hyperedges, hyperedges_timestamps):
    '''Split hyperedges according to timestamp. Sort hyperedges according to
    their timestamp and then split into three parts.

    Returns:
    List. A triplet consisting of hyperedges split into three parts.
    '''
    ground_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if year <= 2007]
    train_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if 2008 <= year <= 2013]
    test_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if 2013 <= year]
    ground_hyperedges = [hyperedges[idx] for idx in ground_idcs]
    train_hyperedges = [hyperedges[idx] for idx in train_idcs]
    test_hyperedges = [hyperedges[idx] for idx in test_idcs]
    return ground_hyperedges, train_hyperedges, test_hyperedges

def split_chsim_dataset(hyperedges):
    '''Split hyperedges according to timestamp. Sort hyperedges according to
    their timestamp and then split into three parts.

    Returns:
    List. A triplet consisting of hyperedges split into three parts.
    '''
    total_hyperedges = len(hyperedges)    
    ground = hyperedges[:int(0.2 * total_hyperedges)]
    train = hyperedges[int(0.2 * total_hyperedges): int(0.5 * total_hyperedges)]
    test = hyperedges[:int(0.5 * total_hyperedges):]
    return ground, train, test
