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

