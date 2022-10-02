from matplotlib import collections  as mc
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import xarray as xr

mesh_data = xr.open_dataset('./data/fesom.mesh.diag.nc')
model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

links = 13

# Taking the first "links" connections
edge_0 = mesh_data.edges[0].values[:links]
edge_1 = mesh_data.edges[1].values[:links]
#edge_tri = mesh_data.edge_tri

# First the connection starting points
x_0 = mesh_data.lon[edge_0].values
y_0 = mesh_data.lat[edge_0].values

# Then the connection endpoints
x_1 = mesh_data.lon[edge_1].values
y_1 = mesh_data.lat[edge_1].values

# Organizing the start and ending connection points for visualization
lines = []
for i in range(links):
    lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])

# Color mask to highlight the last line
c = [(0, 1, 0, 1)]*(links-1)
c.append((1, 0, 0, 1))

lc = mc.LineCollection(lines, colors=c, linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
#for i in range(links):
#    plt.annotate(f"start_{i}", (x_0[i], y_0[i]))
#    plt.annotate(f"end_{i}", (x_1[i], y_1[i]))
ax.margins(0.1)
plt.show()
