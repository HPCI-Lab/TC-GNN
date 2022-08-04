import xarray as xr

PATHS = []
with open('path_hdd.ini', 'r') as file:
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
