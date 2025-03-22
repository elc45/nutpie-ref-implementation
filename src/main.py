import warnings
import argparse
import numpy as np
import arviz as az
import bridgestan as bs
from src.sample import sample
from pathlib import Path
import json

# Convert RuntimeWarnings to Exceptions for debugging
warnings.filterwarnings('error', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='Run MCMC sampling')

parser.add_argument('--data', type=str, required=True,
                   help='Path to JSON file containing data for the model')
parser.add_argument('--model', type=str, required=True,
                   help='Name of the Stan model file (without .stan extension)')
parser.add_argument('--n_samples', type=int, required=False, default=2000,
                   help='Number of samples to generate')
parser.add_argument('--n_warmup', type=int, required=False,default=200,
                   help='Number of warmup iterations')
parser.add_argument('--adapt_mass_matrix', type=lambda x: x.lower() in ['true', '1', 'yes', 't'], 
                   required=False, default=False,
                   help='Whether to adapt the mass matrix')
parser.add_argument('--output_path', type=str, required=False, default='trace.nc',
                   help='Output path for the trace')
parser.add_argument('--matrix_adapt_type', type=str, required=False, default='diag',
                   choices=['diag', 'low_rank', 'full'],
                   help='Type of mass matrix adaptation to use')
parser.add_argument('--step_size_params', type=str, required=False,
                   help='Path to JSON file containing step size adaptation parameters')
parser.add_argument('--matrix_adapt_schedule', type=str, required=False, default=None,
                   help='Path to JSON file containing matrix adaptation schedule')

args = parser.parse_args()

output_path = args.output_path
if not output_path.endswith('.nc'):
    raise ValueError("Output path must be to a .nc (NetCDF) file")

model_name = args.model
n_samples = args.n_samples
warmup_iters = args.n_warmup
adapt_mass_matrix = args.adapt_mass_matrix
matrix_adapt_type = args.matrix_adapt_type
data_path = args.data
matrix_adapt_schedule_path = args.matrix_adapt_schedule
step_size_params_path = args.step_size_params

if matrix_adapt_schedule_path:
    with open(matrix_adapt_schedule_path, 'r') as f:
        matrix_adapt_schedule = json.load(f)
else:
    matrix_adapt_schedule = {}

if step_size_params_path:
    with open(step_size_params_path, 'r') as f:
        step_size_params = json.load(f)
else:
    step_size_params = {}

model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
print("Compiling model...")
model = bs.StanModel(str(model_path), data=data_path)
n_params = model.param_num()
param_names = model.param_names()

def U(q):
    return model.log_density(q) * -1

def grad_U(q):
    return model.log_density_gradient(q)[1] * -1

constrainer = model.param_constrain
init_point = np.random.randn(n_params)

warmup_samples, trace, mass_matrices = sample(U, grad_U, epsilon=0.01, 
                            current_q=init_point, 
                            n_samples=n_samples, 
                            n_warmup=warmup_iters, 
                            adapt_mass_matrix=adapt_mass_matrix, 
                            matrix_adapt_type=matrix_adapt_type,
                            constrainer=constrainer,
                            target_accept_rate=0.8,
                            **matrix_adapt_schedule,
                            **step_size_params)

n_matrices = mass_matrices.shape[0]
mass_matrices = mass_matrices.reshape((1, n_matrices, n_params, n_params))
warmup_samples = warmup_samples.reshape((1, warmup_iters, n_params))
trace = trace.reshape((1, n_samples, n_params))

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