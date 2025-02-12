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

parser.add_argument('--model', type=str, required=True,
                   help='Name of the Stan model file (without .stan extension)')
parser.add_argument('--samples', type=int, required=True,
                   help='Number of MCMC samples to generate')
parser.add_argument('--warmup', type=int, required=False,default=200,
                   help='Number of warmup iterations')
parser.add_argument('--adapt_mass_matrix', type=lambda x: x.lower() in ['true', '1', 'yes', 't'], 
                   required=False, default=True,
                   help='Whether to adapt the mass matrix')
parser.add_argument('--output_path', type=str, required=False, default='trace.npy',
                   help='Path to save the trace')

args = parser.parse_args()

model_name = args.model
n_samples = args.samples
warmup_iters = args.warmup
adapt_mass_matrix = args.adapt_mass_matrix

data = '{"D": 2}'

model_path = Path(__file__).parent.parent / 'models' / f'{model_name}.stan'
model = bs.StanModel(str(model_path), data=data)
n_params = model.param_num()

def U(q):
    return model.log_density_gradient(q)[0]

def grad_U(q):
    return model.log_density_gradient(q)[1]

trace, mass_matrix = sample(U, grad_U, epsilon=0.01, current_q=np.random.randn(n_params), n_samples=n_samples, warmup=warmup_iters, adapt_mass_matrix=adapt_mass_matrix)

n_samples, n_params = trace.shape

draws_dict = {}
draws_dict['x'] = trace.reshape((1, n_samples, n_params))
sample_stats_dict = {}
sample_stats_dict['metric'] = mass_matrix

output = az.from_dict(posterior=draws_dict, sample_stats=sample_stats_dict)

output_path = Path.cwd() / args.output_path
output.to_netcdf(str(output_path))