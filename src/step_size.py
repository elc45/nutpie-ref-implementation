import numpy as np

def update_step_size(epsilon: np.float64, epsilon_bar: np.float64, delta: np.float64, H: np.float64, m: int, t0: int, mu: np.float64, gamma: np.float64, k: np.float64, alpha: int, n_alpha: int):
    log_epsilon = np.log(epsilon)
    log_epsilon_bar = np.log(epsilon_bar)

    H = (1 - 1 / (m + t0)) * H + (1 / (m + t0)) * (delta - alpha / n_alpha)
    log_epsilon = mu - (np.sqrt(m) / gamma) * H
    log_epsilon_bar = m ** -k * log_epsilon + (1 - m ** -k) * log_epsilon_bar

    epsilon = np.exp(log_epsilon)
    epsilon_bar = np.exp(log_epsilon_bar)

    return epsilon, epsilon_bar, H
