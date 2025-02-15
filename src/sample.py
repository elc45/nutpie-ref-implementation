from nuts import nuts_draw
from matrix_adaptation import full_matrix_adapt, diag_matrix_adapt, low_rank_matrix_adapt
import numpy as np
from tqdm import tqdm

def sample(U, grad_U, epsilon, current_q, n_samples, n_warmup=1000, adaptation_window=50, adapt_mass_matrix=True, matrix_adapt_type=None):
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
        current_q, current_grad = nuts_draw(U, grad_U, epsilon, current_q, mass_matrix)
        warmup_samples[:, i] = current_q

        if adapt_mass_matrix:
            draws_buffer[:, buffer_idx] = current_q
            grads_buffer[:, buffer_idx] = current_grad
            buffer_idx = (buffer_idx + 1) % adaptation_window
        
            if (i + 1) % adaptation_window == 0 and i > 0:
                if matrix_adapt_type == 'full':
                    mass_matrix = full_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'diag':
                    mass_matrix = diag_matrix_adapt(draws_buffer, grads_buffer)
                elif matrix_adapt_type == 'low_rank':
                    mass_matrix = low_rank_matrix_adapt(draws_buffer, grads_buffer)
                metrics.append(mass_matrix.copy())

    for i in tqdm(range(n_samples), desc="Sampling"):
        current_q, current_grad = nuts_draw(U, grad_U, epsilon, current_q, mass_matrix)
        samples[i] = current_q
    
    metrics = np.array(metrics)
    return warmup_samples, samples, metrics