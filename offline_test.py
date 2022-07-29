import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data_small = xr.open_dataset("./data/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")
data_big = xr.open_dataset("./data/thetao_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")

# <class 'xarray.core.dataset.DataSet'>
print(data_small)
#print(data_big)

# Returns the Dataset size
print(data_small.dims)

# <class 'xarray.core.dataset.DatasetCoordinates'>
print(data_small.coords)

# <class 'xarray.core.dataset.DataVariables'>
print(data_small.data_vars)

# Returns a dictionary containing all the attributes of the Dataset with their values
d = data_small.attrs
for key in d:
    print(key, '::', d[key])

# Both Coordinates and Data variables can give a 'numpy.ndarray' with a '.values' call
print(data_small.time.values)

# The vstack() produces a 2xlen(data_small.lat) matrix where on
# the first column there are the values of .lat and on the second
# the values of .lon
# lat and lon are <class 'xarray.core.dataset.DataArray'>
v = np.vstack((data_small.lat, data_small.lon))

# The array() basically makes an array only with the .lat,
# so the first parameter you pass -> DON'T USE THIS FUNCTION
a = np.array(data_small.lat, data_small.lon)

# If available, this attribute may tell whether the data is unstructured or not
print(data_small.grid)

### Plotting tests
model_lon = data_small.lon.values
model_lat = data_small.lat.values
data_sample = data_small.tos[0,:].values

plt.figure(figsize=(20, 10))
plt.scatter(model_lon, model_lat, s=1, c=data_sample)
plt.colorbar(orientation='horizontal', pad=0.04)
plt.show()

