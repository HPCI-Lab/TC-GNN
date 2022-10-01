from matplotlib import collections  as mc
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import xarray as xr

mesh_data = xr.open_dataset('./data/fesom.mesh.diag.nc')
model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

step = 2

edge_0 = mesh_data.edges[0].values[:step]
edge_1 = mesh_data.edges[1].values[:step]
#edge_tri = mesh_data.edge_tri

x_0 = mesh_data.lon[edge_0].values
y_0 = mesh_data.lat[edge_0].values
x_1 = mesh_data.lon[edge_1].values
y_1 = mesh_data.lat[edge_1].values

# Putting the lines in the correct format for visualization
lines = []
for i in range(step):
    lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])

lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
plt.show()
