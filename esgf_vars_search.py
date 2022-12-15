import xarray as xr
import matplotlib.pyplot as plt

data = xr.open_dataset('pbo_Omon_AWI-CM-1-1-MR_historical_r2i1p1f1_gn_198101-199012.nc')    # example of file

model_lon = data.lon.values
model_lat = data.lat.values

left = 11
right = 113.5
bottom = -48
top = -2
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

plt.figure(figsize=(20, 10))
#plt.scatter(data.lon, data.lat, s=1)                                                                # world
#plt.scatter(data.lon[region_mask], data.lat[region_mask], s=1)                                      # local region
#plt.scatter(data.lon, data.lat, s=1, c=data.pbo[0].values)                                          # world with variable
plt.scatter(data.lon[region_mask], data.lat[region_mask], s=1, c=data.pbo[0].values[region_mask])   # local region with variable
cbar = plt.colorbar(orientation='horizontal', pad=0.05)
attrs = data.pbo.attrs
label = ""
for key in attrs:
    label += key + ': ' + attrs[key] + "\n"

label += str(data.pbo[0].time.values)
cbar.set_label(label)
plt.show()

data.pbo[0].values.size                 # World nodes
data.pbo[0].values[region_mask].size    # Local nodes
