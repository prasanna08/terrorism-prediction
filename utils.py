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
