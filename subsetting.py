import xarray as xr

data_mesh = xr.open_dataset('./data/fesom.mesh.diag.nc')
data_ssh = xr.open_dataset('./data/ssh.fesom.2010.nc')

## Select 1 value from nz and nz1 between the nz=70 and nz1=69 available
tmp = data_mesh.sel(nz=0.0, nz1=-2.5, method='nearest')
#tmp.to_netcdf('./depth_subset.nc', engine='netcdf4')  # The orignal mesh is 8.5GB, this is 3.6GB

## Select 1 timestamp from the 365 available
tmp = data_ssh.sel(time='2010-01-01T23:54:00', method='nearest')
#tmp.to_netcdf('./time_subset.nc', engine='netcdf4')  # The orignal ssh file is 12.9GB, this is 

## Selecting a spatial subset. We need to find the subset, delete the old coordinates and then add the new reduced ones
model_lon = data_mesh.lon.values
model_lat = data_mesh.lat.values

left = -90
right = -10
bottom = 0
top = 26
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

tmp = data_mesh.drop_vars('lon')
tmp = tmp.drop_vars('lat')
tmp = tmp.assign_coords(lon=('space_subset', model_lon[region_mask]))
tmp = tmp.assign_coords(lat=('space_subset', model_lat[region_mask]))
#tmp.to_netcdf('./spatial_subset.nc', engine='netcdf4')  # The orignal mesh is 8.5GB, this is 8.3GB, but we have to see which pieces to keep from the data variables and eventually from the depths. With just 1 nz and nz1 vale it's 3.4GB
