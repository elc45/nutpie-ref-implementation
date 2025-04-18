from src import nuts
from src import matrix_adaptation
import numpy as np
from tqdm import tqdm
from typing import Callable
from src import step_size

def sample(U: Callable, grad_U: Callable, epsilon: np.float64, current_q: np.ndarray, n_samples: int, 
           constrainer: Callable, n_warmup: int = 1000, adapt_mass_matrix: bool = False, early_window: float = 0.3, early_adapt_window: int = 10, 
           late_adapt_window: int = 80, matrix_adapt_type: str | None = None, eigval_cutoff: float = 100, adapt_step_size: bool = True, target_accept_rate: np.float64 = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    adapt_window = early_adapt_window
    dim = len(current_q)
    samples = np.zeros((n_samples, dim))
    warmup_samples = np.zeros((dim, n_warmup))
    inv_mass_matrix = np.eye(dim)

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

        if matrix_adapt_type == 'diag':             
            run_mean = np.zeros(dim)
            run_var = np.zeros(dim)
            run_grad_mean = np.zeros(dim)
            run_grad_var = np.zeros(dim)
    
    metrics = []
    
    for i in tqdm(range(n_warmup), desc="Warmup"):

        current_q, current_grad, alpha, n_alpha = nuts.draw(U, grad_U, epsilon, current_q, inv_mass_matrix)

        if adapt_step_size:
            if i <= n_warmup:
                epsilon, epsilon_bar, H = step_size.update_step_size(epsilon, epsilon_bar, target_accept_rate, H, m = i+1, t0 = t0, mu = mu, gamma = gamma, k = k, alpha = alpha, n_alpha = n_alpha)
            else:
                epsilon = epsilon_bar
        
        warmup_samples[:, i] = constrainer(current_q)

        if adapt_mass_matrix:
            if matrix_adapt_type == 'diag':
                delta = current_q - run_mean
                run_mean += delta / (buffer_idx + 1)
                delta2 = current_q - run_mean
                run_var += (delta * delta2) / (buffer_idx + 1)

                delta = current_grad - run_grad_mean
                run_grad_mean += delta / (buffer_idx + 1)
                delta2 = current_grad - run_grad_mean
                run_grad_var += (delta * delta2) / (buffer_idx + 1)

                if buffer_idx > 2:
                    inv_mass_matrix = np.diag(np.sqrt(run_var / run_grad_var))
                 
            draws_buffer[:, buffer_idx] = current_q
            grads_buffer[:, buffer_idx] = current_grad
            buffer_idx = (buffer_idx + 1) % adapt_window

            if i % adapt_window == 0 and i > 10:
                if matrix_adapt_type == 'full':
                    inv_mass_matrix = matrix_adaptation.full_matrix_adapt(draws_buffer, grads_buffer)

                elif matrix_adapt_type == 'low_rank':
                    inv_mass_matrix = matrix_adaptation.low_rank_matrix_adapt(draws_buffer, grads_buffer, cutoff = eigval_cutoff)

                if i > n_warmup * early_window:
                    adapt_window = late_adapt_window
                    draws_buffer = np.zeros((dim, adapt_window))
                    grads_buffer = np.zeros((dim, adapt_window))
                    buffer_idx = 0

        metrics.append(inv_mass_matrix.copy())

    for i in tqdm(range(n_samples), desc="Sampling"):
        current_q, current_grad, _, _ = nuts.draw(U, grad_U, epsilon, current_q, inv_mass_matrix)
        samples[i] = constrainer(current_q)

    metrics = np.array(metrics)
    return warmup_samples, samples, metrics