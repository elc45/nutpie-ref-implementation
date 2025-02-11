import numpy as np
import bridgestan as bs
from sample import sample
from pathlib import Path
import sys
import argparse

# Convert RuntimeWarnings to Exceptions for debugging
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='Run MCMC sampling')

parser.add_argument('--model', type=str, required=True,
                   help='Name of the Stan model file (without .stan extension)')
parser.add_argument('--samples', type=int, required=True,
                   help='Number of MCMC samples to generate')
parser.add_argument('--warmup', type=int, required=True,
                   help='Number of warmup iterations')
parser.add_argument('--adapt_mass_matrix', type=lambda x: x.lower() in ['true', '1', 'yes', 't'], 
                   required=False, default=True,
                   help='Whether to adapt the mass matrix')

args = parser.parse_args()

model_name = args.model
n_samples = args.samples
warmup_iters = args.warmup
adapt_mass_matrix = args.adapt_mass_matrix

data = '{"D": 2}'

# Get path to model file relative to this script
model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
model = bs.StanModel(str(model_path), data=data)
n_params = model.param_num()

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace, mass_matrix = sample(U, grad_U, epsilon=0.01, current_q=np.random.randn(n_params), n_samples=n_samples, warmup=warmup_iters, adapt_mass_matrix=adapt_mass_matrix)

# Save trace to file
output_path = Path(__file__).parent.parent / 'tests' / 'trace.npy'
np.save(str(output_path), trace)
print(mass_matrix)