import numpy as np

def normalized_laplacian(A):
    D = np.diag(A.sum(axis=1))
    D = np.pow(D, -0.5)
    L = I - np.dot(np.dot(D, A), D)
    return L

def get_hyperedge_embeddings_from_nodes(H, hyperedges, dim, pool='max'):
    A = np.dot(H, H.T)
    L = normalized_laplacian(A)
    evals, evecs = np.linalg.eig(L)
    embeddings = evecs[:, :dim]
    hyperedge_vectors = []
    for hedge in hyperedges:
        hvecs = evecs[list(edge), :]
        hvec = np.max(hvecs, axis=0) if pool=='max' else np.mean(hvecs, axis=0)
        hyperedge_vectors.append(hvec)
    return np.array(hyperedge_vectors)

def get_hyperedge_embeddings_from_dual(H, dim):
    D = np.dot(H.T, H)
    L = normalized_laplacian(D)
    evals, evecs = np.linalg.eig(L)
    hyperedge_vectors = evecs[:, :dim]
    return hyperedge_vectors
