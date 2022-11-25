from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import xarray as xr

mesh_data = xr.open_dataset('./data/fesom.mesh.diag.nc')
model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values

edge_0 = mesh_data.edges[0].values
edge_1 = mesh_data.edges[1].values

# These indexes start from 1 in the dataset, let's bring them to zero
edge_0 -= 1
edge_1 -= 1

# Spatial subset of the italian coast
left = 12
right = 18.6
bottom = 36.5
top = 43
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

x_0 = []
y_0 = []
x_1 = []
y_1 = []
for i in range(edge_0.size):
    if region_mask[edge_0[i]] & region_mask[edge_1[i]]:     # This edge belongs to the area
        # First the connection starting points
        x_0.append(float(model_lon[edge_0[i]]))
        y_0.append(float(model_lat[edge_0[i]]))
        # Then the connection end points
        x_1.append(float(model_lon[edge_1[i]]))
        y_1.append(float(model_lat[edge_1[i]]))

# Organizing the start and ending connection points for visualization
lines = []
for i in range(len(x_0)):
    lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])

# Color mask to highlight the last line
c = [(0, 1, 0, 1)]*(len(x_0)-1)
#c.append((1, 0, 0, 1))
c.append((0, 1, 0, 1))

lc = mc.LineCollection(lines, colors=c, linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
#for i in range(len(x_0)):
#    plt.annotate(f"start_{i}", (x_0[i], y_0[i]))
#    plt.annotate(f"end_{i}", (x_1[i], y_1[i]))
ax.margins(0.1)
plt.show()


### DEBUG SECTION - printing actual coordinates of connections

## Print each link to see their coordinates - everything was as I expected
for i in range(len(x_0)):
    continue
    print(i)
    print("  ", x_0[i], y_0[i])
    print("  ", x_1[i], y_1[i])

## Plot all "meridian crossing lines"
edge_0 = mesh_data.edges[0].values
edge_1 = mesh_data.edges[1].values
edge_0 -= 1
edge_1 -= 1
x_0 = mesh_data.lon[edge_0].values
y_0 = mesh_data.lat[edge_0].values
x_1 = mesh_data.lon[edge_1].values
y_1 = mesh_data.lat[edge_1].values

meridian_lines = []
for i in range(edge_0.size):
    if ((x_0[i]<(-100)) & (x_1[i]>100)):
        meridian_lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])
    elif ((x_0[i]>100) & (x_1[i]<(-100))):
        meridian_lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])
    else:
        continue

for i in meridian_lines:
    continue
    print(i, "\n")

lc = mc.LineCollection(meridian_lines[::50], linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.margins(0.1)
plt.show()

### END OF DEBUG SECTION
