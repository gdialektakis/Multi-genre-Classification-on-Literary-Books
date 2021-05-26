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