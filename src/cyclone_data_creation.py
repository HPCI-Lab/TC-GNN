import numpy as np
import xarray as xr

mesh_data = xr.open_dataset('../data/fesom.mesh.diag.nc')
cyclones_data = xr.open_dataset('../data/IBTrACS/IBTrACS.since1980.v04r00.nc')


### DELETE USELESS FIELDS ###

vars_keys = mesh_data.data_vars
for key in vars_keys:
    if key != 'lat' and key != 'lon' and key != 'edges':
        mesh_data = mesh_data.drop_vars(key)

mesh_data = mesh_data.drop_vars('nz')   # These 2 are coordinates, not variables
mesh_data = mesh_data.drop_vars('nz1')


### SET THE CYCLONE BASIN ### - TODO: at the moment is just the area of interest of Shishir's work

model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

left = -70
right = 30
bottom = -60
top = -20
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)


### EDGES PROCESSING ###

# Decrease by 1 because for compatibility with Fortran the indexes start from 1, not from 0
edge_0 = mesh_data.edges[0].values
edge_1 = mesh_data.edges[1].values
edge_0 -= 1
edge_1 -= 1

# Saving just the edges in the target area
edges_subset = []
for i in range(len(edge_0)):
    if region_mask[edge_0[i]] & region_mask[edge_1[i]]:
        edges_subset.append([edge_0[i], edge_1[i]])

edges_subset = np.array(edges_subset, dtype="int32")
mesh_data = mesh_data.drop_vars('edges')
mesh_data['edges'] = (('edges_subset', 'n2'), edges_subset)


### NODES PROCESSING ### - just adds 3MB to the pivot file. TODO: you could also just derive the nodes from the set of edges(see check_nodes_edges() for details)

nodes_subset = []
for i in range(len(region_mask)):
    if region_mask[i]:
        nodes_subset.append(i)

nodes_subset = np.array(nodes_subset, dtype="int32")
mesh_data['nodes'] =(('nodes_subset'), nodes_subset)


### DEBUG FUNCTION ###

# Check if the connection nodes and the nodes subset contain exactly the same set of nodes:
def check_nodes_edges():
    edge_0 = edges_subset[:, 0]                     # Take first column
    edge_1 = edges_subset[:, 1]                     # Take second column
    join_sides = np.concatenate((edge_0, edge_1))   # Merge them
    join_sides = np.unique(join_sides)              # Pick unique elements
    join_sides.sort()                               # Sort the lists
    nodes_subset.sort()
    res = True
    counter = 0
    for i in range(len(join_sides)):                # Assuming join_sides and nodes_subset have the same length
        if join_sides[i] != nodes_subset[i]:
            res = False
        else:
            counter += 1                            # This is a double-check to ensure that the verification process was effective
    if res & (counter == len(join_sides)):
        print(f"Edges and nodes contain the same {len(join_sides)} and {len(nodes_subset)} nodes")
    else:
        print("Something went really bad, need to debug")


# Write the pilot mesh to the filesystem - TODO: write some "long_name" describing edges and nodes fields
#mesh_data.to_netcdf('./pilot_mesh_cyclones.nc', engine='netcdf4') # The orignal mesh is 8.5GB, this is 156MB
