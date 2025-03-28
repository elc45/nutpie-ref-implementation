Usage: `python -m src.main [arguments]`

Required arguments:
- `--model`: Path to a Stan model file (without extension)
- `--data`: Path to a data JSON file (without extension)

Optional arguments:
- `--n_samples`: Number of MCMC samples to generate (default: `2000`)
- `--n_warmup`: Number of warmup iterations (default: `200`)
- `--adapt_mass_matrix`: Bool (default: `true`)
- `--output_path`: Path to save the trace as an ArviZ `InferenceData` object in a NetCDF file (default: `trace.nc`)
- `--matrix_adapt_options`: Path to a JSON file with more granular options for mass matrix adaptation, with entries for:
        `early_adapt_window`: frequency with which matrix is updated in early phase
        `late_adapt_window`: frequency with which matrix is updated in regular phase
        `early_window`: fraction of warmup time to be spent in the early adaptation regime
- `--step_size_params`: Path to a JSON file with more granular options for step size adaptation (dual averaging), with entries for:
        `target_accept_rate`
        `gamma`: adaptation regularization scale (default: 0.05)
        `k`: adaptation relaxation exponent (default: 0.75)
        `t0`: adaptation iteration offset (default: 10)
