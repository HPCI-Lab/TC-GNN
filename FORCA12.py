import xarray as xr
import matplotlib.pyplot as plt

PATHS = []
with open('paths.ini', 'r') as file:
    for line in file:
        PATHS.append(line.strip())

data_ssh = xr.open_dataset(PATHS[0], engine='netcdf4')
data_unod = xr.open_dataset(PATHS[1], engine='netcdf4')
data_vnod = xr.open_dataset(PATHS[2], engine='netcdf4')

data_mesh = xr.open_dataset(PATHS[3], engine='netcdf4')

d = data_ssh.attrs
for key in d:
    print(key, '::', d[key])

print()
d = data_ssh.data_vars
for key in d:
    print(key, '::', d[key].long_name)

### Plotting test(in this script the data I'm handling has values and mesh in separate files)
model_lon = data_mesh.lon.values
model_lat = data_mesh.lat.values

# The data is the sea surface height(ssh) for the first day(0) of year 2010
data_sample = data_ssh.ssh[0].values

step = 10
plt.figure(figsize=(20, 10))
plt.scatter(model_lon[::step], model_lat[::step], s=1, c=data_sample[::step])
plt.colorbar(orientation='horizontal', pad=0.04)
plt.savefig('world_FORCA12.png')

# Extraction of a region like in 'offline_test.py'. Mediterranean sea here
left = -10
right = 40
bottom = 30
top = 47
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

step = 1
plt.figure(figsize=(20, 10))
plt.scatter(model_lon[region_mask][::step], model_lat[region_mask][::step], s=1, c=data_sample[region_mask][::step])
cbar = plt.colorbar(orientation='horizontal', pad=0.05)
cbar.set_label(data_ssh.data_vars['ssh'].description)
plt.savefig('mediterranean_FORCA12.png')

# Extraction of a series of depths from the unod file and mesh
model_depth = data_mesh.nz1.values          # could also retrieve from the unod file itself

top = 0
bottom = -50       # negative since it's below the sea level
depth_mask = (model_depth > bottom) & (model_depth < top)

data_sample = data_unod.unod[0][depth_mask][0]  # the last index is used to select one of the depth levels that
                                                # the depth_mask identified

