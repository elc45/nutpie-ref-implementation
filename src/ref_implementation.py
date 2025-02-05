import numpy as np

def leapfrog(q: np.ndarray, p: np.ndarray, epsilon, grad_U):
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * p
    p = p - 0.5 * epsilon * grad_U(q)
    return q, p

def nuts_draw(U, grad_U, epsilon, current_q):
    """
    No-U-Turn Sampler (NUTS) implementation
    
    Parameters:
        U: callable - potential energy function (negative log probability)
        grad_U: callable - gradient of potential energy function
        epsilon: float - step size
        current_q: array - current position
    """
    # Initialize momentum and position
    q = current_q.copy()
    p = np.random.randn(len(q))
    
    # Initial Hamiltonian
    H0 = U(q) + 0.5 * np.sum(p**2)
    
    # Initialize trajectory tree
    q_minus = q.copy()
    q_plus = q.copy() 
    p_minus = p.copy()
    p_plus = p.copy()
    j = 0
    n = 1
    s = 1
    
    # Build trajectory until U-turn or max depth
    while s == 1:
        # Choose direction
        v = 2 * (np.random.rand() < 0.5) - 1
        # Choose subtree to expand
        if v == -1:
            # Expand left subtree
            q_minus, p_minus, _, _, q_prime, n_prime, s_prime = build_tree(
                q_minus, p_minus, -1, j, epsilon, U, grad_U, H0)
        else:
            # Expand right subtree
            q_plus, p_plus, _, _, q_prime, n_prime, s_prime = build_tree(
                q_plus, p_plus, 1, j, epsilon, U, grad_U, H0)
        
        # Update number of valid points and stopping criterion
        n += n_prime
        s = s_prime * (np.dot(q_plus - q_minus, p_minus) >= 0) * \
            (np.dot(q_plus - q_minus, p_plus) >= 0)
        
        # Increment depth
        j += 1
        
        # Break if max depth exceeded
        if j >= 10:
            print("Max depth exceeded")
            break
    
        # Acceptance criterion (Metropolis-Hastings)
    H_new = U(q_prime) + 0.5 * np.sum(p**2)

    delta_H = H_new - H0
    accept_prob = np.exp(delta_H)

    if np.random.rand() < accept_prob:
        return q_prime
    else:
        return current_q

def build_tree(q, p, v, j, epsilon, U, grad_U, H0):
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
        # Single leapfrog step
        p_new, q_new = leapfrog(q, p, epsilon, grad_U)
        H_new = U(q_new) + 0.5 * np.sum(p_new**2)
        
        # Check validity of point (within delta_max of initial energy)
        n_valid = int((H_new - H0) > -1000)
        s_valid = int((H_new - H0) > -100)
        
        return q_new, p_new, q_new, p_new, q_new, n_valid, s_valid
        
    else:
        # Recursively build left and right subtrees
        q_new, p_new, q_minus, p_minus, q_propose, n_propose, s_propose = build_tree(
            q, p, v, j-1, epsilon, U, grad_U, H0)
            
        if s_propose == 1:
            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime = build_tree(
                    q_minus, p_minus, v, j-1, epsilon, U, grad_U, H0)
            else:
                q_new, p_new, _, _, q_prime, n_prime, s_prime = build_tree(
                    q_new, p_new, v, j-1, epsilon, U, grad_U, H0)
                    
            # Update proposal with certain probability
            if np.random.rand() < n_prime/(n_propose + n_prime):
                q_propose = q_prime.copy()
                
            # Update stopping criterion
            s_propose = s_prime * (np.dot(q_new - q_minus, p_minus) >= 0) * \
                       (np.dot(q_new - q_minus, p_new) >= 0)
            
            # Update number of valid points
            n_propose = n_propose + n_prime
            
        return q_new, p_new, q_minus, p_minus, q_propose, n_propose, s_propose

def sample(U, grad_U, epsilon, current_q, n_samples):

    samples = np.zeros((n_samples, len(current_q)))
    
    for i in range(n_samples):
        samples[i] = nuts_draw(U, grad_U, epsilon, current_q)
    return samples