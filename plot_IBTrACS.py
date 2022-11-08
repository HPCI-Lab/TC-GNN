import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd  # geopandas for earth contours

data = xr.open_dataset('./IBTrACS.ALL.v04r00.nc')
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(20, 10))

# Plot every country
#countries.plot(color='grey', ax=ax)

# Plot single continents/countries
#countries[countries["continent"] == "Asia"].plot(color='grey', ax=ax)
countries[countries["name"] == "India"].plot(color='grey', ax=ax)
countries[countries["continent"] == "Africa"].plot(color='grey', ax=ax)

plt.scatter(data.lon[0].values, data.lat[0].values)
plt.show()
