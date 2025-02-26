import numpy as np

def full_matrix_adapt(draw_matrix, grad_matrix):
    draw_cov = np.cov(draw_matrix)
    grad_cov = np.cov(grad_matrix)
    try:
        inv_grad_cov = np.linalg.inv(grad_cov)
    except np.linalg.LinAlgError:
        inv_grad_cov = np.linalg.pinv(grad_cov)
        print(grad_cov)
    Sigma = spdm(draw_cov, inv_grad_cov)
    return Sigma

def low_rank_matrix_adapt(draw_matrix, grad_matrix, gamma=1e-5, cutoff=0.01):

    draws_normalized = (draw_matrix - draw_matrix.mean(axis=0)[:, np.newaxis]) / draw_matrix.std(axis=0)[:, np.newaxis]
    grads_normalized = (grad_matrix - grad_matrix.mean(axis=0)[:, np.newaxis]) / grad_matrix.std(axis=0)[:, np.newaxis]
    U_draw, _, _ = np.linalg.svd(draws_normalized, full_matrices=False)
    U_grad, _, _ = np.linalg.svd(grads_normalized, full_matrices=False)

    S = np.hstack([U_draw, U_grad])

    Q, _ = np.linalg.qr(S)

    P_draw = Q.T @ draw_matrix
    P_grad = Q.T @ grad_matrix

    C_draw = P_draw @ P_draw.T + gamma * np.eye(Q.shape[1])
    C_grad = P_grad @ P_grad.T + gamma * np.eye(Q.shape[1])

    Sigma = spdm(C_draw, C_grad)

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    indices = np.where(eigvals >= cutoff)[0]
    U_selected = eigvecs[:, indices]

    mass_matrix = Q @ U_selected

    return mass_matrix

def diag_matrix_adapt(draw_matrix, grad_matrix):
    draw_variance = np.var(draw_matrix, axis=1)
    grad_variance = np.var(grad_matrix, axis=1)

    scaling_factors = np.sqrt(draw_variance / grad_variance)

    mass_matrix = np.diag(scaling_factors)
    return mass_matrix

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
    eigvals_A = np.maximum(eigvals_A, 1e-10)

    A_sqrt = eigvecs_A @ np.diag(np.sqrt(eigvals_A)) @ eigvecs_A.T
    A_inv_sqrt = eigvecs_A @ np.diag(1 / np.sqrt(eigvals_A)) @ eigvecs_A.T

    M = A_sqrt @ B @ A_sqrt

    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    eigvals_M = np.maximum(eigvals_M, 1e-10)
    M_sqrt = eigvecs_M @ np.diag(eigvals_M) @ eigvecs_M.T

    spdm_matrix = A_inv_sqrt @ M_sqrt @ A_inv_sqrt
    
    return spdm_matrix