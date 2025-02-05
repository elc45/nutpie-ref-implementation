import bridgestan as bs
import numpy as np
from ref_implementation import sample

data = '{"D": 1}'

model = bs.StanModel("../models/normal.stan",data=data)

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace = sample(U, grad_U, epsilon=0.01, current_q=np.array([0.1]), n_samples=1000)

# Save trace to file
np.save('tests/trace.npy', trace)