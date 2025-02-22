import numpy as np
from tqdm import tqdm

def leapfrog(q: np.ndarray, p: np.ndarray, epsilon: np.float64, grad_U, inv_mass_matrix: np.ndarray):
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * np.dot(inv_mass_matrix, p)
    p = p - 0.5 * epsilon * grad_U(q)
    return q, p

def sample_hmc(U, grad_U, epsilon, q_init, n_samples, L=10):
    dim = len(q_init)
    samples = np.zeros((n_samples, dim))
    proposed_samples = np.zeros((n_samples, L, dim))
    mass_matrix = np.eye(dim)
    q = q_init.copy()
    p = np.random.multivariate_normal(np.zeros_like(q_init), mass_matrix)

    for i in tqdm(range(n_samples), desc="Sampling"):
        q_propose = q.copy()
        p_propose = p.copy()

        for j in range(L):
            q_propose, p_propose = leapfrog(q_propose, p_propose, epsilon, grad_U, mass_matrix)
        
        H_old = U(q) + 0.5 * np.dot(p, p)
        H_new = U(q_propose) + 0.5 * np.dot(p_propose, p_propose)
        q = q_propose if np.random.rand() < np.exp(H_old - H_new) else q
        samples[i,:] = q

        p = np.random.multivariate_normal(np.zeros_like(q), mass_matrix)

    return samples, proposed_samples