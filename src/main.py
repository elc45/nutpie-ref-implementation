import numpy as np
import bridgestan as bs
from sample import sample
from pathlib import Path
import sys

if len(sys.argv) != 3:
    print("Usage: python main.py <model_name> <n_samples>")
    sys.exit(1)

model_name = sys.argv[1]
n_samples = int(sys.argv[2])

data = '{"D": 1}'

# Get path to model file relative to this script
model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
model = bs.StanModel(str(model_path), data=data)

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace = sample(U, grad_U, epsilon=0.01, current_q=np.array([0.4]), n_samples=n_samples)

# Save trace to file
output_path = Path(__file__).parent.parent / 'tests' / 'trace.npy'
np.save(str(output_path), trace)