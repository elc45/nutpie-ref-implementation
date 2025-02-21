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

def test_tree_building():
    """Test the tree building process of NUTS by examining a simple case"""
    
    def U(q):
        return 0.5 * q**2  # Simple 1D quadratic potential
    
    def grad_U(q):
        return q
    
    # Initial conditions
    q0 = np.array([1.0])
    p0 = np.array([1.0])
    epsilon = 0.1
    mass_matrix = np.array([1.0])
    
    # Build a tree of depth 2 manually to test each step
    q = q0
    p = p0
    u = np.random.uniform()
    v = 1 if u < 0.5 else -1
    
    # First doubling - depth 1
    q_fwd, p_fwd = leapfrog(q, p, epsilon, grad_U, mass_matrix)
    q_bwd, p_bwd = leapfrog(q, p, -epsilon, grad_U, mass_matrix)
    
    print(q_fwd, q_bwd, p_fwd, p_bwd, p0)
    # Store tree array for visualization
    tree = np.array([
        [q_bwd, q0, q_fwd],
        [p_bwd, p0, p_fwd]
    ])
    print("\nTree at depth 1:")
    print("Position states:", tree[0])
    print("Momentum states:", tree[1])
    
    # Check energy conservation approximately holds
    initial_energy = U(q0) + 0.5 * p0**2
    fwd_energy = U(q_fwd) + 0.5 * p_fwd**2
    bwd_energy = U(q_bwd) + 0.5 * p_bwd**2
    
    assert np.abs(initial_energy - fwd_energy) < 0.1, "Energy should be approximately conserved"
    assert np.abs(initial_energy - bwd_energy) < 0.1, "Energy should be approximately conserved"
    
    # Second doubling - depth 2
    if v == 1:
        q_fwd2, p_fwd2 = leapfrog(q_fwd, p_fwd, epsilon, grad_U, mass_matrix)
        tree = np.array([
            [q_bwd, q0, q_fwd, q_fwd2],
            [p_bwd, p0, p_fwd, p_fwd2]
        ])
        print("\nTree at depth 2 (forward):")
        print("Position states:", tree[0])
        print("Momentum states:", tree[1])
        assert np.abs(U(q_fwd2) + 0.5 * p_fwd2**2 - initial_energy) < 0.2, "Energy conservation should hold for depth 2"
    else:
        q_bwd2, p_bwd2 = leapfrog(q_bwd, p_bwd, -epsilon, grad_U, mass_matrix)
        tree = np.array([
            [q_bwd2, q_bwd, q0, q_fwd],
            [p_bwd2, p_bwd, p0, p_fwd]
        ])
        print("\nTree at depth 2 (backward):")
        print("Position states:", tree[0])
        print("Momentum states:", tree[1])
        assert np.abs(U(q_bwd2) + 0.5 * p_bwd2**2 - initial_energy) < 0.2, "Energy conservation should hold for depth 2"

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

def test_leapfrog():
    def U(q):
        return 0.5 * q**2
    
    def grad_U(q):
        return q

    # Initial conditions
    q0 = -3.57286962
    p0 = 0.78
    epsilon = 0.01
    n_steps = 10
    
    t = n_steps * epsilon
    q_true = q0 * np.cos(t) + p0 * np.sin(t)
    p_true = p0 * np.cos(t) - q0 * np.sin(t)
    
    # Use the leapfrog function instead of manual implementation
    q = np.array([q0])
    p = np.array([p0])
    
    mass_matrix = np.array([1.0])  # For 1D case
    
    for i in range(n_steps):
        q, p = leapfrog(q, p, epsilon, grad_U, mass_matrix)
        print("p:", p)
        print("p_true:", p0 * np.cos(t) - q0 * np.sin(i * epsilon))
    
    # print(f"Numerical solution: q={q}, p={p}")
    # print(f"Analytical solution: q={q_true}, p={p_true}")
    # print(f"Relative error: q={abs((q-q_true)/q_true)}, p={abs((p-p_true)/p_true)}")
    
    initial_energy = 0.5 * p0**2 + U(q0)
    final_energy = 0.5 * p**2 + U(q)
    print(f"Initial energy: {initial_energy}, Final energy: {final_energy}")
    # Check if numerical solution approximately matches analytical solution
    assert np.abs(q - q_true) < 0.01, "Position from leapfrog should approximate analytical solution"
    assert np.abs(p - p_true) < 0.01, "Momentum from leapfrog should approximate analytical solution"
    
    # Check energy conservation
    
    rel_energy_error = abs((final_energy - initial_energy)/initial_energy)
    print(f"Relative energy error: {rel_energy_error}")
    assert rel_energy_error < 1e-4, "Energy should be approximately conserved"

test_leapfrog()