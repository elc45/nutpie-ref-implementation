import numpy as np
from typing import Callable

def leapfrog(q: np.ndarray, p: np.ndarray, epsilon: np.float64, grad_U: Callable, inv_mass_matrix: np.ndarray):
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * np.dot(inv_mass_matrix, p)
    p = p - 0.5 * epsilon * grad_U(q)
    return q, p

def draw(U: Callable, grad_U: Callable, epsilon: np.float64, current_q: np.ndarray, inv_mass_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    No-U-Turn Sampler (NUTS) implementation
    
    Parameters:
        U: callable - potential energy function (negative log probability)
        grad_U: callable - gradient of potential energy function
        epsilon: float - step size
        current_q: array - current position
        mass_matrix: array - mass matrix
    """

    if inv_mass_matrix is None:
        inv_mass_matrix = np.eye(len(current_q))

    q = current_q.copy()
    p = np.random.multivariate_normal(np.zeros_like(q), cov=np.eye(len(q)))
    
    q_minus = q.copy()
    q_plus = q.copy() 
    p_minus = p.copy()
    p_plus = p.copy()
    j = 0
    n = 1
    s = 1
    u = np.random.uniform(0, np.exp(-U(q) - 0.5 * np.dot(p, p)))
    FORWARD = 1
    BACKWARD = -1

    while s == 1:
        v = np.random.choice([FORWARD, BACKWARD])
        if v == BACKWARD:
            q_minus, p_minus, _, _, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(
                q_minus, p_minus, u, BACKWARD, j, epsilon, U, grad_U, inv_mass_matrix, current_q, p)
        else:
            _, _, q_plus, p_plus, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(
                q_plus, p_plus, u, FORWARD, j, epsilon, U, grad_U, inv_mass_matrix, current_q, p)
        
        if s_prime == 1:
            current_q = q_prime if np.random.rand() < n_prime / n else current_q

        n += n_prime
        s = s_prime * (np.dot(q_plus - q_minus, p_minus) >= 0) * \
        (np.dot(q_plus - q_minus, p_plus) >= 0)
        
        j += 1

    final_grad = grad_U(current_q)
    return current_q, final_grad, alpha, n_alpha

def build_tree(q, p, u, v, j, epsilon, U, grad_U, inv_mass_matrix, q0, p0):
    """
    Build a binary tree of states by recursively doubling trajectory length.
    
    Parameters:
        q: array - position
        p: array - momentum 
        v: int - direction (-1 or 1)
        j: int - iteration/depth
        epsilon: float - step size
        U: callable - potential energy function
        grad_U: callable - gradient of potential energy
        H0: float - initial value of Hamiltonian
        q0: array - initial position
        p0: array - initial momentum
        
    Returns:
        q_minus: array - leftmost position
        p_minus: array - leftmost momentum
        q_plus: array - rightmost position
        p_plus: array - rightmost momentum
        q_propose: array - proposed new position
        n_propose: int - number of valid points in the subtree
        s_propose: int - whether the subtree is valid (1) or not (0)
        alpha_prime: float - sum of alpha values in the subtree
        n_alpha_prime: int - number of alpha values in the subtree
    """
    if j == 0:
        q_new, p_new = leapfrog(q, p, epsilon, grad_U, inv_mass_matrix)
        
        L = -U(q_new) - 0.5 * np.dot(p_new, p_new)
        
        n_valid = int(u < np.exp(L))
        s_valid = int(L - np.log(u) > -1000)
        
        return q_new, p_new, q_new, p_new, q_new, n_valid, s_valid, min(1, np.exp(L + U(q0) + 0.5 * np.dot(p0,p0))), 1
        
    else:
        q_minus, p_minus, q_plus, p_plus, q_propose, n_valid, s_propose, alpha_prime, n_alpha_prime = build_tree(
            q, p, u, v, j-1, epsilon, U, grad_U, inv_mass_matrix, q0, p0)
            
        if s_propose == 1:
            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_double_prime, s_prime, alpha_double_prime, n_alpha_double_prime = build_tree(
                    q_minus, p_minus, u, v, j-1, epsilon, U, grad_U, inv_mass_matrix, q0, p0)
            else:
                _, _, q_plus, p_plus, q_prime, n_double_prime, s_prime, alpha_double_prime, n_alpha_double_prime = build_tree(
                    q_plus, p_plus, u, v, j-1, epsilon, U, grad_U, inv_mass_matrix, q0, p0)

            if n_valid + n_double_prime > 0:
                if np.random.rand() < n_double_prime/(n_valid + n_double_prime):
                    q_propose = q_prime.copy()
            
            alpha_prime = alpha_prime + alpha_double_prime
            n_alpha_prime = n_alpha_prime + n_alpha_double_prime

            s_propose = s_prime * (np.dot(q_plus - q_minus, p_minus) >= 0) * \
                       (np.dot(q_plus - q_minus, p_plus) >= 0)
            
            n_valid += n_double_prime
            
        return q_minus, p_minus, q_plus, p_plus, q_propose, n_valid, s_propose, alpha_prime, n_alpha_prime