import json
import warnings
import argparse
import numpy as np
import arviz as az
import bridgestan as bs
from sample import sample
from pathlib import Path

# Convert RuntimeWarnings to Exceptions for debugging
warnings.filterwarnings('error', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='Run MCMC sampling')

parser.add_argument('--data', type=str, required=True,
                   help='Path to JSON file containing data for the model')
parser.add_argument('--model', type=str, required=True,
                   help='Name of the Stan model file (without .stan extension)')
parser.add_argument('--n_samples', type=int, required=True,
                   help='Number of samples to generate')
parser.add_argument('--n_warmup', type=int, required=False,default=200,
                   help='Number of warmup iterations')
parser.add_argument('--adapt_mass_matrix', type=lambda x: x.lower() in ['true', '1', 'yes', 't'], 
                   required=False, default=False,
                   help='Whether to adapt the mass matrix')
parser.add_argument('--output_path', type=str, required=False, default='trace.npy',
                   help='Output path for the trace')
parser.add_argument('--matrix_adapt_type', type=str, required=False, default='diag',
                   choices=['diag', 'low_rank', 'full'],
                   help='Type of mass matrix adaptation to use')

args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
warmup_iters = args.n_warmup
adapt_mass_matrix = args.adapt_mass_matrix
matrix_adapt_type = args.matrix_adapt_type
data_path = args.data

model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
print("Compiling model...")
model = bs.StanModel(str(model_path), data=data_path)
n_params = model.param_num()
param_names = model.param_names()

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

constrainer = model.param_constrain
init_point = np.random.randn(n_params)

warmup_samples, trace, mass_matrices = sample(U, grad_U, epsilon=0.01, 
                            current_q=init_point, 
                            n_samples=n_samples, 
                            n_warmup=warmup_iters, 
                            adapt_mass_matrix=adapt_mass_matrix, 
                            matrix_adapt_type=matrix_adapt_type,
                            constrainer=constrainer)

n_matrices = mass_matrices.shape[0]
mass_matrices = mass_matrices.reshape((1, n_matrices, n_params, n_params))
warmup_samples = warmup_samples.reshape((1, warmup_iters, n_params))
trace = trace.reshape((1, n_samples, n_params))

import matplotlib.pyplot as plt

# Create a figure with subplots for each parameter
n_params = len(param_names)
fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
if n_params == 1:
    axes = [axes]

# Plot histogram for each parameter
for i, (param, ax) in enumerate(zip(param_names, axes)):
    param_samples = trace[0, :, i]
    ax.hist(param_samples, bins=50, density=True)
    ax.set_title(f'Histogram of {param}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

draws_dict = {}
for param in param_names:
    param_idx = param_names.index(param)
    draws_dict[param] = trace[:,:,param_idx]

warmup_dict = {}
for param in param_names:
    param_idx = param_names.index(param)
    warmup_dict[param] = warmup_samples[:,:,param_idx]

sample_stats_dict = {}
sample_stats_dict['metric'] = mass_matrices

output = az.from_dict(
    posterior=draws_dict, 
    sample_stats=sample_stats_dict,
    warmup_posterior=warmup_dict,
    save_warmup=True
)

output_path = Path.cwd() / args.output_path
output_path = str(output_path)
output.to_netcdf(output_path)
print("Trace saved to ", output_path)