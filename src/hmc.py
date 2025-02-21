import numpy as np
from tqdm import tqdm

def leapfrog(q: np.ndarray, p: np.ndarray, epsilon: np.float64, grad_U, inv_mass_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * inv_mass_matrix @ p
    p = p - 0.5 * epsilon * grad_U(q)
    return q, p

q = np.random.randint(0, 10, size=1)
p = np.random.randint(0, 10, size=1)

def sample_hmc(U, grad_U, epsilon, q, n_samples, n_warmup=1000):
    dim = len(q)
    samples = np.zeros((n_samples, dim))
    warmup_samples = np.zeros((dim, n_warmup))
    mass_matrix = np.eye(dim)

    for i in tqdm(range(n_warmup), desc="Warmup"):
        q_propose, _ = leapfrog(q, p, epsilon, grad_U, mass_matrix)
        q_new = q_propose if np.random.rand() < np.exp(U(q_propose) - U(q)) else q
        warmup_samples[:, i] = q_new
        p = np.random.multivariate_normal(np.zeros_like(q), mass_matrix)

    for i in tqdm(range(n_samples), desc="Sampling"):
        q_propose, _ = leapfrog(q, p, epsilon, grad_U, mass_matrix)
        q_new = q_propose if np.random.rand() < np.exp(U(q_propose) - U(q)) else q
        samples[i] = q_new
        p = np.random.multivariate_normal(np.zeros_like(q), mass_matrix)

    return warmup_samples, samples