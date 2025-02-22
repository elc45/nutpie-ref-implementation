import numpy as np
import bridgestan as bs
from src.hmc import sample_hmc
from src.sample import sample
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats

model_name = 'normal'
n_samples = 25000

data_path = Path(__file__).parent / 'data' / 'norm_constr.json'
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

_, trace, _ = sample(U, grad_U, epsilon=0.1, 
                            current_q=init_point, 
                            n_samples=n_samples,
                            constrainer=constrainer)

trace = trace.reshape((1, n_samples, n_params))

# Print number of unique values in trace
for i, param in enumerate(param_names):
    param_samples = trace[0, :, i]
    n_unique = len(np.unique(param_samples))
    print(f"Number of unique values for {param}: {n_unique}")


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