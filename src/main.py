import numpy as np
import bridgestan as bs
from sample import sample
from pathlib import Path
import sys

if len(sys.argv) != 4:
    print("Usage: python main.py <model_name> <n_samples> <warmup_iters>")
    sys.exit(1)

model_name = sys.argv[1]
n_samples = int(sys.argv[2])
warmup_iters = int(sys.argv[3])

data = '{"D": 1}'

# Get path to model file relative to this script
model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
model = bs.StanModel(str(model_path), data=data)
n_params = model.param_num()

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace, mass_matrix = sample(U, grad_U, epsilon=0.01, current_q=np.random.randn(n_params), n_samples=n_samples, warmup=warmup_iters)

# Save trace to file
output_path = Path(__file__).parent.parent / 'tests' / 'trace.npy'
np.save(str(output_path), trace)
print(mass_matrix)