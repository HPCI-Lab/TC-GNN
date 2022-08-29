import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data_mesh = xr.open_dataset('./data/fesom.mesh.diag.nc')

model_lon = data_small.lon.values
model_lat = data_small.lat.values

left = -90
right = -10
bottom = 0
top = 26
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

