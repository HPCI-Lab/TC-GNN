from pyexpat import model
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


d = model_lon[elements].max(axis=1) - model_lon[elements].min(axis=1)       # BROKEN - NEED 'elements'
no_cyclic_elem = np.argwhere(d < 100).ravel()                               # BROKEN - NEED 'd'
triang = mtri.Triangulation(model_lon, model_lat, elements[no_cyclic_elem]) # BROKEN - NEED 'elements' and 'no_cyclic_elem'


##### Deeper levels of FESOM data - ALL BROKEN DOWN HERE

# Taking the first date value and the 3150m depth data
data_at_3150_m = dataset.thetao[0, 34].values

# Data not available at 3150m is signed as 'nan'
data_at_3150_m

### Naive approach - interpolating considering all triangles:
bad_interpolation = mtri.LinearTriInterpolator

### Better approach - tell matplotlib to ignore the triangles with 'nan' values