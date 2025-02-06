from ref_implementation import nuts_draw
import numpy as np
from tqdm import tqdm
import sys

def sample(U, grad_U, epsilon, current_q, n_samples):

    samples = np.zeros((n_samples, len(current_q)))
    
    for i in tqdm(range(n_samples)):
        current_q = nuts_draw(U, grad_U, epsilon, current_q)
        samples[i] = current_q
        
    return samples