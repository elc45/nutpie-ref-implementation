from src import nuts
from src import matrix_adaptation
import numpy as np
from tqdm import tqdm
from typing import Callable

def sample(U: Callable, grad_U: Callable, epsilon: np.float64, current_q: np.ndarray, n_samples: int, constrainer: Callable, n_warmup: int = 1000, adaptation_window: int = 50, adapt_mass_matrix: bool = False, matrix_adapt_type: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = len(current_q)
    samples = np.zeros((n_samples, dim))
    warmup_samples = np.zeros((dim, n_warmup))
    mass_matrix = np.eye(dim)
    
    if adapt_mass_matrix:
        draws_buffer = np.zeros((dim, adaptation_window))
        grads_buffer = np.zeros((dim, adaptation_window))
        buffer_idx = 0
    
    metrics = []

    for i in tqdm(range(n_warmup), desc="Warmup"):
        current_q, current_grad = nuts.draw(U, grad_U, epsilon, current_q, mass_matrix)
        warmup_samples[:, i] = constrainer(current_q)

        if adapt_mass_matrix:
            draws_buffer[:, buffer_idx] = current_q
            grads_buffer[:, buffer_idx] = current_grad
            buffer_idx = (buffer_idx + 1) % adaptation_window
        
            if (i + 1) % adaptation_window == 0 and i > 0:
                if matrix_adapt_type == 'full':
                    mass_matrix = matrix_adaptation.full_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'diag':
                    mass_matrix = matrix_adaptation.diag_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'low_rank':
                    mass_matrix = matrix_adaptation.low_rank_matrix_adapt(draws_buffer, grads_buffer)
        
        metrics.append(mass_matrix.copy())

    for i in tqdm(range(n_samples), desc="Sampling"):
        current_q, current_grad = nuts.draw(U, grad_U, epsilon, current_q, mass_matrix)
        samples[i] = constrainer(current_q)

    metrics = np.array(metrics)
    return warmup_samples, samples, metrics