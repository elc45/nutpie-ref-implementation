from src import nuts
from src import matrix_adaptation
import numpy as np
from tqdm import tqdm
from typing import Callable
from src import step_size

def sample(U: Callable, grad_U: Callable, epsilon: np.float64, target_accept_rate: np.float64, current_q: np.ndarray, n_samples: int, constrainer: Callable, n_warmup: int = 1000, adapt_window: int = 50, adapt_mass_matrix: bool = False, matrix_adapt_type: str | None = None, adapt_step_size: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = len(current_q)
    samples = np.zeros((n_samples, dim))
    warmup_samples = np.zeros((dim, n_warmup))
    mass_matrix = np.eye(dim)

    H = 0
    gamma = 0.05
    k = 0.75
    mu = np.log(10 * epsilon)
    t0 = 10
    epsilon_bar = 1

    if adapt_mass_matrix:
        draws_buffer = np.zeros((dim, adapt_window))
        grads_buffer = np.zeros((dim, adapt_window))
        buffer_idx = 0
    
    metrics = []

    for i in tqdm(range(n_warmup), desc="Warmup"):

        current_q, current_grad, alpha, n_alpha = nuts.draw(U, grad_U, epsilon, current_q, mass_matrix)

        if adapt_step_size:
            if i <= n_warmup:
                epsilon, epsilon_bar, H = step_size.update_step_size(epsilon, epsilon_bar, target_accept_rate, H, m = i+1, t0 = t0, mu = mu, gamma = gamma, k = k, alpha = alpha, n_alpha = n_alpha)
            else:
                epsilon = epsilon_bar
        
        warmup_samples[:, i] = constrainer(current_q)

        if adapt_mass_matrix:
            draws_buffer[:, buffer_idx] = current_q
            grads_buffer[:, buffer_idx] = current_grad
            buffer_idx = (buffer_idx + 1) % adapt_window
            if i % adapt_window == 0 and i > 200:
                if matrix_adapt_type == 'full':
                    mass_matrix = matrix_adaptation.full_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'diag':
                    mass_matrix = matrix_adaptation.diag_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'low_rank':
                    mass_matrix = matrix_adaptation.low_rank_matrix_adapt(draws_buffer, grads_buffer)
        
        metrics.append(mass_matrix.copy())
        # mass_matrix = np.eye(dim)

    for i in tqdm(range(n_samples), desc="Sampling"):
        current_q, current_grad, _, _ = nuts.draw(U, grad_U, epsilon, current_q, mass_matrix)
        samples[i] = constrainer(current_q)

    metrics = np.array(metrics)
    return warmup_samples, samples, metrics