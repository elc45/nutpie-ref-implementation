import numpy as np

def full_matrix_adapt(draw_matrix, grad_matrix):

    cov = draw_matrix @ draw_matrix.T
    diag = np.sqrt(np.diag(cov))
    draw_cov = cov / np.outer(diag, diag)

    cov = grad_matrix @ grad_matrix.T
    diag = np.sqrt(np.diag(cov))
    grad_cov = cov / np.outer(diag, diag)
    inv_grad_cov = np.linalg.inv(grad_cov)
    Sigma = spdm(draw_cov, inv_grad_cov)

    return Sigma

def low_rank_matrix_adapt(draw_matrix, grad_matrix):
    pass

def diag_matrix_adapt(draw_matrix, grad_matrix):
    draw_variance = np.var(draw_matrix, axis=1)
    grad_variance = np.var(grad_matrix, axis=1)

    scaling_factors = np.sqrt(draw_variance / grad_variance)

    return np.diag(scaling_factors)

def spdm(A, B):
    """
    Compute the symmetric positive definite mean (SPDM) of matrices A and B.
    
    Parameters:
        A: np.ndarray, shape (p, p)
        B: np.ndarray, shape (p, p)
    
    Returns:
        spdm_matrix: np.ndarray, shape (p, p)
            Symmetric positive definite mean of A and B.
    """

    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    A_sqrt = eigvecs_A @ np.diag(np.sqrt(eigvals_A)) @ eigvecs_A.T
    A_inv_sqrt = eigvecs_A @ np.diag(1 / np.sqrt(eigvals_A)) @ eigvecs_A.T

    M = A_sqrt @ B @ A_sqrt

    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    M_sqrt = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.T

    spdm_matrix = A_inv_sqrt @ M_sqrt @ A_inv_sqrt
    return spdm_matrix