Usage: `python src/main.py [arguments]`

Required arguments:
- `--model`: Name of the Stan model file (without .stan extension)
- `--samples`: Number of MCMC samples to generate

Optional arguments:
- `--warmup`: Number of warmup iterations (default: `200`)
- `--adapt_mass_matrix`: Whether to adapt the mass matrix during warmup (default: `true`)
- `--output_path`: Path to save the trace file (default: `trace.nc`)