#!/bin/bash

### CLUSTER ENVIRONMENT SETUP STEPS:
module load miniconda
python -m venv env_HPC
source ./env_HPC/bin/activate
pip install --upgrade pip
pip install xarray
pip install netcdf4

