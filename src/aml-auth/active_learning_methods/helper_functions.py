import numpy as np
import scipy.sparse as sp
from numpy import ones


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def create_random_pool_and_initial_sets(X, y, n_samples_for_intial):
    training_indices = np.random.choice(range(X.shape[0]), size=n_samples_for_intial, replace=False)

    X_train = X[training_indices]
    y_train = y[training_indices]

    X_pool = delete_rows_csr(X, training_indices)
    y_pool = np.delete(y, training_indices)

    return X_train, y_train, X_pool, y_pool