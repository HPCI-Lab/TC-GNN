#!/bin/bash

### CLUSTER ENVIRONMENT SETUP STEPS:
#module load miniconda
python3 -m venv env_HPC
source ./env_HPC/bin/activate
pip install --upgrade pip
pip install xarray
pip install netcdf4
pip install matplotlib
pip install dask

## PyTorch setup:

pip install torch
# Ensure at least PyTorch 1.11.0 and CUDA are installed:
# python -c "import torch; print(torch.__version__)"	# 1.12.1+cu102
# python -c "import torch; print(torch.version.cuda)"	# 10.2

# The following torch and cuda versions are the ones verified in the previous step
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install torch-geometric

# Optional(remember to change the torch and cuda version) dependencies:
# pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
