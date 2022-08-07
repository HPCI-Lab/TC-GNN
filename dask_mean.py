import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

PATHS = []
with open('path_hdd.ini', 'r') as file:
    for line in file:
        PATHS.append(line.strip())

file_list_ssh = []
file_list_ssh.append(PATHS[0])
horizontal_split = 100000
vertical_split = 1
time_split = 100

# Open multiple files as a single dataset, requires dask to be installed.
# W.r.t. xr.open_dataset, this method appears to be loading the "ssh" variable as a dask.array rather than a
# "simple" array
ssh = xr.open_mfdataset(PATHS[0], chunks={"time": time_split, "nod2": horizontal_split, "nz1": vertical_split})

# Computation of mean
ssh_mean = ssh.mean(dim="time")
ssh_mean.compute()