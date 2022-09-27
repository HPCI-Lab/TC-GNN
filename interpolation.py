import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time_func as tm

PATHS = []
with open('paths.ini', 'r') as file:
    for line in file:
        PATHS.append(line.strip())

# Opening the sea surface height data with its mesh
data_ssh = xr.open_dataset(PATHS[0], engine='netcdf4')
data_mesh = xr.open_dataset(PATHS[3], engine='netcdf4')

##### Advanced interpolation, taking triangles into account(only FESOM, since we need data on vertices)

model_lon = data_mesh.lon.values
model_lat = data_mesh.lat.values
data_sample = data_ssh.ssh[0].values

# Each row in the following variable is a triangle, and each value in the row is the index of vertices
elements = (data_mesh.elements.data.astype('int32') - 1).T

#elements[0, :]

# Accessing longitude and latitude coordinates with the elements indexes
model_lon[elements[0]]
model_lat[elements[0]]

# Since tracer data in FESOM2 is located on the vertices of triangles, we can do this to get the values for the first triangle:
data_sample[elements[0]]

# And the mean value for the triangle would be:
data_sample[elements[0]].mean()

# In a matrix, axis=0 is the vertical while axis=1 is the horizontal
d = model_lon[elements].max(axis=1) - model_lon[elements].min(axis=1)

# argwhere finds the indices of non-zero values and ravel() puts it in 1D form
no_cyclic_elem = np.argwhere(d < 100).ravel()

# Create a matplotlib triangulation based of only non cyclic triangles
triang = mtri.Triangulation(model_lon, model_lat, elements[no_cyclic_elem])

# RUN IT ON THE CLUSTER, AS IT TAKES A LOT OF TIME AND RAM
interpolation = mtri.LinearTriInterpolator(triang, data_sample)

# Let's save the unstructured and interpolated grid in pictures
plt.figure(figsize=(20, 10))
plt.scatter(model_lon[::20], model_lat[::20], s=1, c=data_sample[::20])
plt.show()
plt.savefig('unstructured.png')

##### Let's try something different

# Taking the first date value ssh data
#data_at_20100101 = data_ssh.ssh[0].values
