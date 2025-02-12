import numpy as np

def nutpie_update(draw_matrix, grad_matrix, gamma=1e-5, cutoff=0.01):
    """
    Perform low-rank mass matrix update using the Nutpie algorithm.
    
    Parameters:
        draw_matrix: np.ndarray, shape (p, n)
            Matrix of normalized draws in p-dimensional parameter space.
        grad_matrix: np.ndarray, shape (p, n)
            Matrix of normalized gradients in p-dimensional parameter space.
        gamma: float, optional
            Regularization parameter to add to the diagonal.
        cutoff: float, optional
            Eigenvalue cutoff for dimensionality reduction.
    
    Returns:
        mass_matrix: np.ndarray, shape (p, k)
            Low-rank approximation to the mass matrix, where k is determined by cutoff.
    """

    U_draw, _, _ = np.linalg.svd(draw_matrix, full_matrices=False)
    U_grad, _, _ = np.linalg.svd(grad_matrix, full_matrices=False)

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