import numpy as np

def svd_compress(gray, k=50):
    U, S, Vt = np.linalg.svd(gray, full_matrices=False)
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :])).astype(np.uint8)
