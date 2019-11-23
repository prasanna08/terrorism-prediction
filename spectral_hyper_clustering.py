import numpy as np
import scipy.sparse as sp

def normalized_laplacian(A):
    D = sp.csr_matrix(np.diag(np.power(A.toarray().sum(axis=1), -0.5)))
    L = sp.eye(D.shape[0]) - D * A * D
    return L

def get_hyperedge_embeddings_from_nodes(H, hyperedges, dim, pool='max'):
    A = H * H.T
    L = normalized_laplacian(A)
    evals, evecs = sp.linalg.eigsh(L, k=dim, which='SM')
    embeddings = evecs
    hyperedge_vectors = []
    for hedge in hyperedges:
        hvecs = evecs[list(hedge), :]
        hvec = np.max(hvecs, axis=0) if pool=='max' else np.mean(hvecs, axis=0)
        hyperedge_vectors.append(hvec)
    return np.array(hyperedge_vectors)

def get_hyperedge_embeddings_from_dual(H, dim):
    D = H.T * H
    L = normalized_laplacian(D)
    evals, evecs = sp.linalg.eigsh(L, k=dim, which='SM')
    hyperedge_vectors = evecs
    return hyperedge_vectors
