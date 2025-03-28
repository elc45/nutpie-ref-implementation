import numpy as np

def full_matrix_adapt(draw_matrix, grad_matrix):

    draw_means = draw_matrix.mean(axis=1)
    grad_means = grad_matrix.mean(axis=1)
    
    draws_rescaled = (draw_matrix - draw_means[:, None])
    grads_rescaled = (grad_matrix - grad_means[:, None])
    
    draw_cov = np.cov(draws_rescaled)
    grad_cov = np.cov(grads_rescaled)
        
    Sigma = spdm(draw_cov, grad_cov)
    inv_mass_matrix = np.linalg.inv(Sigma)
    return inv_mass_matrix

def low_rank_matrix_adapt(draw_matrix, grad_matrix, gamma=1e-5, cutoff=100):

    d, n_draws = draw_matrix.shape

    draw_means = draw_matrix.mean(axis=1)
    grad_means = grad_matrix.mean(axis=1)
    
    draw_stds = np.std(draw_matrix, axis=1)
    grad_stds = np.std(grad_matrix, axis=1)
    
    stds = np.sqrt(draw_stds / grad_stds)
    
    draw_scales = 1.0 / (stds * n_draws)
    grad_scales = stds / n_draws
    
    draws_rescaled = (draw_matrix - draw_means[:, None]) * draw_scales[:, None]
    grads_rescaled = (grad_matrix - grad_means[:, None]) * grad_scales[:, None]

    U_draw, _, _ = np.linalg.svd(draws_rescaled, full_matrices=False)
    U_grad, _, _ = np.linalg.svd(grads_rescaled, full_matrices=False)

    S = np.hstack([U_draw, U_grad])

    Q, _ = np.linalg.qr(S)

    P_draw = Q.T @ draw_matrix
    P_grad = Q.T @ grad_matrix

    C_draw = P_draw @ P_draw.T + gamma * np.eye(d)
    C_grad = P_grad @ P_grad.T + gamma * np.eye(d)

    Sigma = spdm(C_draw, C_grad)

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    indices = np.where((eigvals > cutoff) | (eigvals < (1/cutoff)))[0]
    U_selected = eigvecs[:, indices]

    U = Q @ U_selected
    mass_matrix = U @ np.diag(eigvals[indices] - 1) @ U.T + np.eye(d)
    inv_mass_matrix = np.linalg.inv(mass_matrix)
    return inv_mass_matrix

def diag_matrix_adapt(draw_matrix, grad_matrix):
    draw_variance = np.var(draw_matrix, axis=1)
    grad_variance = np.var(grad_matrix, axis=1)

    scaling_factors = np.sqrt(draw_variance / grad_variance)

    inv_mass_matrix = np.diag(scaling_factors)
    return inv_mass_matrix

def spdm(A, B):

    eigvals_B, eigvecs_B = np.linalg.eig(B)

    B_sqrt = eigvecs_B @ np.diag(np.sqrt(eigvals_B)) @ eigvecs_B.T
    B_inv_sqrt = eigvecs_B @ np.diag(1 / np.sqrt(eigvals_B)) @ eigvecs_B.T

    M = B_sqrt @ A @ B_sqrt

    eigvals_M, eigvecs_M = np.linalg.eig(M)
    M_sqrt = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.T

    spdm_matrix = B_inv_sqrt @ M_sqrt @ B_inv_sqrt
    
    return spdm_matrix