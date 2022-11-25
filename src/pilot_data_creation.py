import xarray as xr

PATH = ''

mesh_data = xr.open_dataset('../data/fesom.mesh.diag.nc')
data_ssh = xr.open_dataset('../data/ssh.fesom.2010.nc')
data_unod = xr.open_dataset(PATH + 'unod.fesom.2010.nc', engine='netcdf4')
data_vnod = xr.open_dataset(PATH + 'vnod.fesom.2010.nc', engine='netcdf4')


### Process the mesh file ###

# Extracting longitude and latitude values of the mesh
model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

# Spatial subset of the Mediterranean sea, but to use it you'll need to fix bug #1
left = -10
right = 40
bottom = 30
top = 47
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

# We'll use the first 100000 values of longitude and latitude to avoid the problems related to bug #1
subset = 100000

# Changing the lon and lat values in the dataset
mesh_data = mesh_data.drop_vars('lon')
mesh_data = mesh_data.drop_vars('lat')
mesh_data = mesh_data.assign_coords(lon=('spatial_subset', model_lon[:subset])) # could also be 'nodes_subset'
mesh_data = mesh_data.assign_coords(lat=('spatial_subset', model_lat[:subset]))

# Taking just the "surface" middle level depth, dropping the depth levels
mesh_data = mesh_data.drop_vars('nz')
mesh_data = mesh_data.sel(nz1=0.0, method='nearest')

mesh_data = mesh_data.drop_vars('edge_cross_dxdy')
mesh_data = mesh_data.drop_vars('edge_tri')
mesh_data = mesh_data.drop_vars('elem_area')
mesh_data = mesh_data.drop_vars('elem_part')
mesh_data = mesh_data.drop_vars('gradient_sca_x')
mesh_data = mesh_data.drop_vars('gradient_sca_y')
mesh_data = mesh_data.drop_vars('nlevels')
mesh_data = mesh_data.drop_vars('nlevels_nod2D')
mesh_data = mesh_data.drop_vars('nod_area')
mesh_data = mesh_data.drop_vars('nod_in_elem2D')
mesh_data = mesh_data.drop_vars('nod_in_elem2D_num')
mesh_data = mesh_data.drop_vars('nod_part')
mesh_data = mesh_data.drop_vars('nodes')
mesh_data = mesh_data.drop_vars('zbar_e_bottom')
mesh_data = mesh_data.drop_vars('zbar_n_bottom')

### ELEMENTS PROCESSING ###
"""
# Each row in the following variable is a triangle, and each value in the row is the index of a vertex
elements = (mesh_data.elements.data.astype('int32') - 1).T
elem_mask = []

# Removing the triangles that have at least a vertex missing from the spatial subset
for v1, v2, v3 in elements:
    #res = (region_mask[v1] & region_mask[v2] & region_mask[v3])    # bug #1
    res = (v1 < subset) & (v2 < subset) & (v3 < subset)
    elem_mask.append(res)   # append True or False depending on the triangle vertices

mesh_data = mesh_data.drop_vars('elements')
mesh_data['elements'] = (('elem_subset', 'n3'), elements[elem_mask])
"""

### EDGES PROCESSING ###

edges = (mesh_data.edges.data.astype('int32') - 1).T
edges_mask = []

# Removing edges outside of the subset
for start, end in edges:        # TODO: this is the much slower version, for the faster one look at cyclone_data_creation.py
    res = (start < subset) & (end < subset)
    edges_mask.append(res)

mesh_data = mesh_data.drop_vars('edges')
mesh_data['edges'] = (('edge_subset', 'n2'), edges[edges_mask])

# Write the pilot mesh to the filesystem
#mesh_data.to_netcdf('./pilot_mesh.nc', engine='netcdf4') # The orignal mesh is 8.5GB, this is 3.9MB


### Process the data files ###

data_ssh = data_ssh.sel(time='2010-01-01T00:00:00', method='nearest')
data_unod = data_unod.sel(nz1=0.0, method='nearest')
data_unod = data_unod.sel(time='2010-01-01T00:00:00', method='nearest')
data_vnod = data_vnod.sel(nz1=0.0, method='nearest')
data_vnod = data_vnod.sel(time='2010-01-01T00:00:00', method='nearest')

data = data_ssh.ssh
#data.to_netcdf('./pilot_ssh.nc', engine='netcdf4')
data = data_unod.unod
#data.to_netcdf('./pilot_unod.nc', engine='netcdf4')
data = data_vnod.vnod
#data.to_netcdf('./pilot_vnod.nc', engine='netcdf4')
