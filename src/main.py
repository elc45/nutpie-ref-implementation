import bridgestan as bs
import numpy as np
from ref_implementation import sample
from pathlib import Path

data = '{"D": 1}'

# Get path to model file relative to this script
model_path = Path(__file__).parent.parent / 'models' / 'normal.stan'
model = bs.StanModel(str(model_path), data=data)

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace = sample(U, grad_U, epsilon=0.01, current_q=np.array([0.4]), n_samples=10000)

# Save trace to file
output_path = Path(__file__).parent.parent / 'tests' / 'trace.npy'
np.save(str(output_path), trace)