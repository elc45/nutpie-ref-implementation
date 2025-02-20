import numpy as np
import mypy

def leapfrog(q: np.ndarray, p: np.ndarray, epsilon: np.float64, grad_U, inv_mass_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * inv_mass_matrix @ p
    p = p - 0.5 * epsilon * grad_U(q)
    return q, p

def nuts_draw(U, grad_U, epsilon: np.float64, current_q: np.ndarray, mass_matrix: np.ndarray | None = None, max_depth: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    No-U-Turn Sampler (NUTS) implementation
    
    Parameters:
        U: callable - potential energy function (negative log probability)
        grad_U: callable - gradient of potential energy function
        epsilon: float - step size
        current_q: array - current position
        mass_matrix: array - mass matrix
    """

    if mass_matrix is None:
        mass_matrix = np.eye(len(current_q))

    inv_mass_matrix = np.linalg.inv(mass_matrix)

    q = current_q.copy()
    p = np.random.multivariate_normal(np.zeros_like(q), mass_matrix)
    
    H0 = U(q) + 0.5 * np.sum(p**2)
    
    q_minus = q.copy()
    q_plus = q.copy() 
    p_minus = p.copy()
    p_plus = p.copy()
    j = 0
    n = 1
    s = 1
    FORWARD = 1
    BACKWARD = -1

    while s == 1:
        v = np.random.choice([FORWARD, BACKWARD])
        if v == BACKWARD:
            q_minus, p_minus, _, _, q_prime, n_prime, s_prime = build_tree(
                q_minus, p_minus, BACKWARD, j, epsilon, U, grad_U, H0, inv_mass_matrix)
        else:
            q_plus, p_plus, _, _, q_prime, n_prime, s_prime = build_tree(
                q_plus, p_plus, FORWARD, j, epsilon, U, grad_U, H0, inv_mass_matrix)
        
        n += n_prime
        s = s_prime * (np.dot(q_plus - q_minus, p_minus) >= 0) * \
            (np.dot(q_plus - q_minus, p_plus) >= 0)
        
        j += 1
        
        if j > max_depth:
            break

    H_new = U(q_prime) + 0.5 * np.sum(p**2)

    delta_H = H_new - H0
    accept_prob = np.exp(delta_H)

    current_q = q_prime if np.random.rand() < accept_prob else current_q
    final_grad = grad_U(current_q)
    
    return current_q, final_grad

def build_tree(q, p, v, j, epsilon, U, grad_U, H0, inv_mass_matrix):
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
        
    Returns:
        q: array - final position
        p: array - final momentum
        q_minus: array - leftmost position
        p_minus: array - leftmost momentum
        q_propose: array - proposed new position
        n_propose: int - number of valid points in the subtree
        s_propose: int - whether the subtree is valid (1) or not (0)
    """
    if j == 0:
        p_new, q_new = leapfrog(q, p, epsilon, grad_U, inv_mass_matrix)

        H_new = U(q_new) + 0.5 * np.sum(p_new**2)
        
        n_valid = int((H_new - H0) > -1000)
        if not n_valid:
            print("divergence")
        s_valid = int((H_new - H0) > -100)
        
        return q_new, p_new, q_new, p_new, q_new, n_valid, s_valid
        
    else:
        q_new, p_new, q_minus, p_minus, q_propose, n_propose, s_propose = build_tree(
            q, p, v, j-1, epsilon, U, grad_U, H0, inv_mass_matrix)
            
        if s_propose == 1:
            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime = build_tree(
                    q_minus, p_minus, v, j-1, epsilon, U, grad_U, H0, inv_mass_matrix)
            else:
                q_new, p_new, _, _, q_prime, n_prime, s_prime = build_tree(
                    q_new, p_new, v, j-1, epsilon, U, grad_U, H0, inv_mass_matrix)
                    
            if np.random.rand() < n_prime/(n_propose + n_prime):
                q_propose = q_prime.copy()
                
            s_propose = s_prime * (np.dot(q_new - q_minus, p_minus) >= 0) * \
                       (np.dot(q_new - q_minus, p_new) >= 0)
            
            n_propose = n_propose + n_prime
            
        return q_new, p_new, q_minus, p_minus, q_propose, n_propose, s_propose