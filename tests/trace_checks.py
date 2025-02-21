import numpy as np
import bridgestan as bs
from src.sample import sample
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats

model_name = 'normal'
n_samples = 10000
warmup_iters = 1000
adapt_mass_matrix = False
matrix_adapt_type = None

data_path = Path(__file__).parent / 'data' / 'none.json'
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
init_point = -5 * np.random.randn(n_params)

warmup_samples, trace, mass_matrices = sample(U, grad_U, epsilon=0.001, 
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

# Print mean for each parameter
print("\nParameter means:")
for i, param in enumerate(param_names):
    param_samples = trace[0, :, i]
    print(f"{param}: {np.mean(param_samples):.3f}")


# Plot trace plots for each parameter
# Create a figure for trace plots
fig, ax = plt.subplots(1, 1, figsize=(10, 3))

# Plot trace for the parameter
param_samples = trace[0, :, 0]
ax.plot(param_samples)
ax.set_title(f'Trace plot of {param_names[0]}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')


plt.tight_layout()
plt.show()
