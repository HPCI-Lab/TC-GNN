import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Both datasets are AWI-ESM, low resolution historical
# variable_id: tos; ocean surface temperature
data_small = xr.open_dataset("./data/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")
# variable_id: thetao; ocean potential temperature
data_big = xr.open_dataset("./data/thetao_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001-185012.nc")

# <class 'xarray.core.dataset.DataSet'>
print(data_small)

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
#plt.savefig('world_fesom.png')

# Now, cut a smaller region, say, the Barbados
left = -90
right = -10
bottom = 0
top = 26
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

step = 1
plt.figure(figsize=(20, 10))
plt.scatter(model_lon[region_mask][::step], model_lat[region_mask][::step], s=30, c=data_sample[region_mask][::step])
plt.colorbar(orientation='horizontal', pad=0.04)
plt.show()
#plt.savefig('barbados_fesom.png')


# "Big" dataset plot:
model_lon = data_big.lon.values
model_lat = data_big.lat.values
data_sample = data_big.thetao[5][10].values

plt.figure(figsize=(20, 10))
plt.scatter(model_lon, model_lat, s=1, c=data_sample)
cbar = plt.colorbar(orientation='horizontal', pad=0.04)
cbar.set_label(data_big.data_vars['thetao'].standard_name)
plt.show()
#plt.savefig('world_fesom.png')
