Usage: `python -m src.main [arguments]`

Required arguments:
- `--model`: Name of the Stan model file (without .stan extension)
- `--data`: Number of MCMC samples to generate

Optional arguments:
- `--n_samples`: Number of samples (default: `2000`)
- `--n_warmup`: Number of warmup iterations (default: `200`)
- `--adapt_mass_matrix`: Bool (default: `true`)
- `--output_path`: Path to save the trace as an ArviZ `InferenceData` object in a NetCDF file (default: `trace.nc`)
- `--matrix_adapt_options`: Path to a JSON file with more granular options for mass matrix adaptation, with entries for:
        `early_adapt_window`: frequency with which matrix is updated in early phase
        `late_adapt_window`: frequency with which matrix is updated in regular phase
        `early_window`: fraction of warmup time to be spent in the early adaptation regime