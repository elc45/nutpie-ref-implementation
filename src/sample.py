from nuts import nuts_draw
from matrix_adaptation import full_matrix_adapt
import numpy as np
from tqdm import tqdm

def sample(U, grad_U, epsilon, current_q, n_samples, warmup=1000, adaptation_window=50, adapt_mass_matrix=True):
    dim = len(current_q)
    samples = np.zeros((n_samples, dim))
    mass_matrix = np.eye(dim)
    
    if adapt_mass_matrix:
        draws_buffer = np.zeros((dim, adaptation_window))
        grads_buffer = np.zeros((dim, adaptation_window))
        buffer_idx = 0
    
    for i in tqdm(range(warmup)):
        current_q, current_grad = nuts_draw(U, grad_U, epsilon, current_q, mass_matrix)
        
        if adapt_mass_matrix:
            draws_buffer[:, buffer_idx] = current_q
            grads_buffer[:, buffer_idx] = current_grad
            buffer_idx = (buffer_idx + 1) % adaptation_window
        
            if (i + 1) % adaptation_window == 0 and i > 0:
                mass_matrix = full_matrix_adapt(draws_buffer, grads_buffer)
    
    for i in tqdm(range(n_samples)):
        current_q, current_grad = nuts_draw(U, grad_U, epsilon, current_q, mass_matrix)
        samples[i] = current_q
        
    return samples, mass_matrix