import xarray as xr

data_mesh = xr.open_dataset('./data/fesom.mesh.diag.nc')
data_ssh = xr.open_dataset('./data/ssh.fesom.2010.nc')

# Extracting longitude and latitude values of the mesh
model_lon = data_mesh.lon.values
model_lat = data_mesh.lat.values

# The area we're gonna keep is the Mediterranean sea
left = -10
right = 40
bottom = 30
top = 47
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

# Changing the lon and lat values in the dataset
data_mesh = data_mesh.drop_vars('lon')
data_mesh = data_mesh.drop_vars('lat')
data_mesh = data_mesh.assign_coords(lon=('spatial_subset', model_lon[region_mask])) # could also be 'nodes_subset'
data_mesh = data_mesh.assign_coords(lat=('spatial_subset', model_lat[region_mask]))

# Taking just the surface depth value, dropping the useless values
data_mesh = data_mesh.drop_vars('nz1')
data_mesh = data_mesh.sel(nz=0.0, method='nearest')

data_mesh = data_mesh.drop_vars('edge_cross_dxdy')
data_mesh = data_mesh.drop_vars('edge_tri')
data_mesh = data_mesh.drop_vars('edges')
data_mesh = data_mesh.drop_vars('elem_area')
data_mesh = data_mesh.drop_vars('elem_part')
data_mesh = data_mesh.drop_vars('gradient_sca_x')
data_mesh = data_mesh.drop_vars('gradient_sca_y')
data_mesh = data_mesh.drop_vars('nlevels')
data_mesh = data_mesh.drop_vars('nlevels_nod2D')
data_mesh = data_mesh.drop_vars('nod_area')
data_mesh = data_mesh.drop_vars('nod_in_elem2D')
data_mesh = data_mesh.drop_vars('nod_in_elem2D_num')
data_mesh = data_mesh.drop_vars('nod_part')
data_mesh = data_mesh.drop_vars('nodes')
data_mesh = data_mesh.drop_vars('zbar_e_bottom')
data_mesh = data_mesh.drop_vars('zbar_n_bottom')

# Each row in the following variable is a triangle, and each value in the row is the index of a vertex
elements = (data_mesh.elements.data.astype('int32') - 1).T
elem_mask = []

# Removing the triangles that have at least a vertex missing from the spatial subset
for v1, v2, v3 in elements:
    res = (region_mask[v1] & region_mask[v2] & region_mask[v3])
    if res:
        elem_mask.append(True)
    else:
        elem_mask.append(False)
    # TODO: filter also the edge data if the edge data ends up being useful

data_mesh = data_mesh.drop_vars('elements')
data_mesh['elements'] = (('elem_subset', 'nz3'), elements[elem_mask])

#data_mesh.to_netcdf('./pilot_data.nc', engine='netcdf4') # The orignal mesh is 8.5GB, this is 337M
