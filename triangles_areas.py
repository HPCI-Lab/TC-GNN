import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Plot of the triangles areas to see how dense the mesh gets

mesh_data = xr.open_dataset('./data/fesom.mesh.diag.nc')

model_lon = mesh_data.lon.values
model_lat = mesh_data.lat.values
elements = (mesh_data.elements.data.astype('int32') - 1).T
centroids_lon = []
centroids_lat = []
areas = []

for v1, v2, v3 in elements:
    Ox = (model_lon[v1] + model_lon[v2] + model_lon[v3])/3
    Oy = (model_lat[v1] + model_lat[v2] + model_lat[v3])/3
    area = (1/2)*abs(model_lon[v1]*(model_lat[v2]-model_lat[v3]) + model_lon[v2]*(model_lat[v3]-model_lat[v1]) + model_lon[v3]*(model_lat[v1]-model_lat[v2]))
    centroids_lon.append(Ox)
    centroids_lat.append(Oy)
    areas.append(area)

centroids_lon = np.array(centroids_lon)
centroids_lat = np.array(centroids_lat)
areas = np.array(areas)
data_sample = mesh_data.elem_area.values

left = -10
right = 40
bottom = 30
top = 47
region_mask = (centroids_lon > left) & (centroids_lon < right) & (centroids_lat < top) & (centroids_lat > bottom)

step = 30
plt.figure(figsize=(20, 10))
#plt.scatter(centroids_lon[region_mask][::step], centroids_lat[region_mask][::step], s=1, c=areas[region_mask][::step])
plt.scatter(centroids_lon[::step], centroids_lat[::step], s=1, c=data_sample[::step])
plt.colorbar(orientation='horizontal', pad=0.04)
plt.show()
