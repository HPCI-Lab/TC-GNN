import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

PATHS = []
with open('paths.ini', 'r') as file:
    for line in file:
        PATHS.append(line.strip())

inputs_list = [PATHS[4]]  # Useful when input data is stored in multiple files
horizontal_split = 100000
vertical_split = 1
time_split = 100

# Open multiple files as a single dataset, requires dask to be installed.
# W.r.t. xr.open_dataset, this method appears to be loading the "ssh" variable as a dask.array rather than a
# "simple" array
dataset = xr.open_mfdataset(inputs_list, chunks={"time": time_split, "nod2": horizontal_split, "nz1": vertical_split})

# Computation of mean
dataset_mean = dataset.mean(dim="time")
computed_mean = dataset_mean.compute()
computed_mean.to_netcdf('./computed_mean.nc', engine='netcdf4')
