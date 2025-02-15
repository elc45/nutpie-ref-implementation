from src.nuts import *
from src.matrix_adaptation import *
import numpy as np

def test_nuts_draw():

    def U(q):
        return 0.5 * np.dot(q, q)  # Standard normal log density (up to constant)
    
    def grad_U(q):
        return q
    
    q0 = np.array([1.0, 1.0])
    mass_matrix = np.eye(2)
    epsilon = 0.1
    
    # Run single draw
    q1, grad1 = nuts_draw(U, grad_U, epsilon, q0, mass_matrix)
    
    assert q1.shape == (2,), "NUTS draw should return vector of correct dimension"
    assert grad1.shape == (2,), "NUTS draw should return gradient of correct dimension"

def test_matrix_adaptation():

    dim = 3
    n_samples = 100
    
    true_cov = np.array([[1.0, 0.5, 0.2],
                        [0.5, 2.0, 0.3],
                        [0.2, 0.3, 1.5]])
    
    samples = np.random.multivariate_normal(mean=np.zeros(dim), 
                                          cov=true_cov, 
                                          size=n_samples).T
    grads = np.random.multivariate_normal(mean=np.zeros(dim),
                                        cov=np.linalg.inv(true_cov),
                                        size=n_samples).T
    
    adapted_full = full_matrix_adapt(samples, grads)
    adapted_diag = diag_matrix_adapt(samples, grads)
    adapted_low_rank = low_rank_matrix_adapt(samples, grads)
    
    assert adapted_full.shape == (dim, dim), "Full matrix adaptation should return square matrix"
    assert adapted_diag.shape == (dim, dim), "Diagonal matrix adaptation should return square matrix"
    assert adapted_low_rank.shape == (dim, dim), "Low rank matrix adaptation should return square matrix"
    
    assert np.allclose(adapted_diag, np.diag(np.diag(adapted_diag)))