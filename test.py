import xarray as xr
import numpy as np

data_small = xr.open_dataset("./data/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")
data_big = xr.open_dataset("./data/thetao_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")

print(data_small)
#print(data_big)

# Lists all the attributes in the Dataset
print(data_small.attrs)

# The vstack() produces a 2xlen(data_small.lat) matrix where on
# the first column there are the values of .lat and on the second
# the values of .lon
v = np.vstack((data_small.lat, data_small.lon))

# The array() basically makes an array only with the .lat,
# so the first parameter you pass -> DON'T USE THIS FUNCTION
a = np.array(data_small.lat, data_small.lon)

# Useful to tell whether the data is unstructured or not
print(data_small.grid)

