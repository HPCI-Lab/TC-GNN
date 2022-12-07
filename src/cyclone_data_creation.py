import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from utils.time_func import start_time, stop_time
from utils.cyclones_func import extract_storms

mesh_data = xr.open_dataset('../data/fesom.mesh.diag.nc')
cyclones_data = xr.open_dataset('../data/IBTrACS/IBTrACS.since1980.v04r00.nc')


### DELETE USELESS FIELDS ###

vars_keys = mesh_data.data_vars
for key in vars_keys:
    if key != 'lat' and key != 'lon' and key != 'edges':
        mesh_data = mesh_data.drop_vars(key)

mesh_data = mesh_data.drop_vars('nz')   # These 2 are coordinates, not variables
mesh_data = mesh_data.drop_vars('nz1')


### SET THE CYCLONE BASIN ### - setting it on South Indian storms on La Reunion longitude and latitude values

model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

left = 11
right = 113.5
bottom = -48
top = -2
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


### NODES PROCESSING ### - usually just adds 3MB to the pilot file. TODO: you could also just derive the nodes from the set of edges(see check_nodes_edges() for details)

nodes_subset = []
for i in range(len(region_mask)):
    if region_mask[i]:
        nodes_subset.append(i)

nodes_subset = np.array(nodes_subset, dtype="int32")
mesh_data['nodes'] =(('nodes_subset'), nodes_subset)


### STORMS EXTRACTION ### - basin of choice: South Indian

storms = extract_storms(cyclones_data, b'SI')


### CYCLONES INTERPOLATION ###

#cyclones_lon = cyclones_data.reunion_lon[1].values[4]      # 1 as the storm to map, 4 as the day where the wind was recorded
#cyclones_lat = cyclones_data.reunion_lat[1].values[4]
cyclones_lon = cyclones_data.reunion_lon[storms].values[# TODO here you need only recorded wind values]
cyclones_lat = cyclones_data.reunion_lat[storms].values[#TODO same as above]

model_lon = mesh_data.lon[mesh_data.nodes].values
model_lat = mesh_data.lat[mesh_data.nodes].values

nodes = []
for m in range(len(model_lon)):
    nodes.append([model_lon[m], model_lat[m]])
nodes = np.array(nodes)

cyclones = []
for c in range(len(cyclones_lon)):
    cyclones.append([cyclones_lon[c], cyclones_lat[c]])
cyclones = np.array(cyclones)

# For each cyclone point, retrive the closest node index
#timestamp = start_time()
#stop_time(timestamp, "points creation")
#node_indexes = np.argmin(np.sum((nodes - cyclones)**2, axis=1))


''' Nikolay's method
data_sample = cyclones_data.reunion_wind[1].values[4]
points = np.vstack((cyclones_lon, cyclones_lat)).T
nn_interpolation = NearestNDInterpolator(points, data_sample)
from scipy.interpolate import NearestNDInterpolator
nn_interpolation = NearestNDInterpolator(points, data_sample)
interpolated_nn_fesom = nn_interpolation((model_lon, model_lat))
plt.imshow(interpolated_nn_fesom)
'''

''' Voronoi - it gets the polygons pretty easily, but then you need something like Kirkpatrick's DAG
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)   # qhull_options='Qbb Qc Qx' put this as parameter together with points to avoid the empty region used as infinite
fig = voronoi_plot_2d(vor)
plt.show()
'''


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
#mesh_data.to_netcdf('./pilot_mesh_cyclones.nc', engine='netcdf4') # The orignal mesh is 8.5GB, this is 155MB
