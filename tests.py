from ref_implementation import *
import numpy as np

def build_sigma(rho):    
    sigma = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            sigma[i,j] = rho**abs(i-j)
    return sigma

cov = build_sigma(0.9)
draws = np.random.multivariate_normal(mean=np.zeros(12), cov=cov, size=50).T
grads = np.random.multivariate_normal(mean=np.zeros(12), cov=np.linalg.inv(cov), size=50).T

test = nutpie_update(draws, grads, gamma=1e-5, cutoff=2)

print(test)