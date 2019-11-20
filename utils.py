def hyperedges_to_incidence_matrix(hyperedges, total_nodes):
    '''Tansfrom a list of hyperedges into incidence matrix hypergraph.

    Args:
    hyperedges: List(frozenset). A list of hyperedges where each hyperedge is
        frozenset of nodes it contains.
    total_nodes: Int. Total number of nodes in hyperedge.

    Returns:
    H: scipy.sparse.csr_matrix. Sparse matrix representation of hyperedges.
    '''
    pass

def incidence_matrix_to_hyperedges(H):
    '''Transfrom incidence matrix representation of hypergraph H into list of
    hyperedges.

    Args:
    H: scipy.sparse.csr_matrix. Sparse matrix representation of hypergraph H.

    Returns:
    hyperedges: List(fronzenset). List of hyperedges where each hyperedge is
        frozenset of nodes it contains.
    '''
    pass

def network_deconvolution(A, threshold):
    '''Deconvolution of a network represented by adjacency matrix A.

    Args:
    A: np.array. Adjacency matrix of the network.
    threshold: float. All edges with weeight below threshold are removed from 
        deconvoluted matrix and the weight for rest is set to 1.

    Returns:
    D: np.array. Deconvoluted adjacency matrix of network.
    '''
    pass
