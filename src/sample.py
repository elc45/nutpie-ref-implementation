from ref_implementation import nuts_draw
from matrix_adaptation import nutpie_update
import numpy as np
from tqdm import tqdm
import sys

def sample(U, grad_U, epsilon, current_q, n_samples, adaptation_window=50):
    dim = len(current_q)
    samples = np.zeros((n_samples, dim))
    mass_matrix = np.eye(dim)  # Start with identity matrix
    
    # Storage for adaptation
    draws_buffer = np.zeros((dim, adaptation_window))
    grads_buffer = np.zeros((dim, adaptation_window))
    buffer_idx = 0
    
    for i in tqdm(range(n_samples)):
        # Get sample and its gradient in one call
        current_q, current_grad = nuts_draw(U, grad_U, epsilon, current_q, mass_matrix)
        
        # Store for adaptation
        draws_buffer[:, buffer_idx] = current_q
        grads_buffer[:, buffer_idx] = current_grad
        buffer_idx = (buffer_idx + 1) % adaptation_window
        
        # Adapt mass matrix every adaptation_window steps
        if (i + 1) % adaptation_window == 0 and i > 0:
            # Normalize the buffers
            draws_centered = draws_buffer - draws_buffer.mean(axis=1, keepdims=True)
            draws_normalized = draws_centered / draws_centered.std(axis=1, keepdims=True)
            
            grads_centered = grads_buffer - grads_buffer.mean(axis=1, keepdims=True)
            grads_normalized = grads_centered / grads_centered.std(axis=1, keepdims=True)
            
            # Update mass matrix
            mass_matrix = nutpie_update(draws_normalized, grads_normalized)
        
        samples[i] = current_q
        
    return samples, mass_matrix