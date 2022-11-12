import geopandas as gpd  # geopandas for earth contours
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

data = xr.open_dataset('./data/IBTrACS.since1980.v04r00.nc')
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(20, 10))

# Plot every country
#countries.plot(color='grey', ax=ax)

# Plot single continents/countries
countries[countries["continent"] == "Asia"].plot(color='grey', ax=ax)
#countries[countries["name"] == "India"].plot(color='grey', ax=ax)
#countries[countries["continent"] == "Africa"].plot(color='grey', ax=ax)

plt.scatter(data.lon[0].values, data.lat[0].values)
plt.show()

# Evaluation of some variables
data = data.drop_vars('numobs')         # no of observations
data = data.drop_vars('sid')            # serial id
data = data.drop_vars('season')         # season when storm started
data = data.drop_vars('number')         # storm number(within season)
data = data.drop_vars('subbasin')       # current sub-basin
data = data.drop_vars('name')           # name of system
data = data.drop_vars('source_bom')     # source data information for this storm for Australian tracks
data = data.drop_vars('source_usa')
data = data.drop_vars('source_jma')
data = data.drop_vars('source_cma')
data = data.drop_vars('source_hko')
data = data.drop_vars('source_new')
data = data.drop_vars('source_reu')
data = data.drop_vars('source_nad')
data = data.drop_vars('source_wel')
data = data.drop_vars('source_td5')
data = data.drop_vars('source_td6')
data = data.drop_vars('source_ds8')
data = data.drop_vars('source_neu')
data = data.drop_vars('source_mlc')
data = data.drop_vars('iso_time')       # time in ISO
data = data.drop_vars('nature')         # nature of the cyclone
data = data.drop_vars('wmo_pres')       # official WMO agency
data = data.drop_vars('track_type')     #

basins = data.basin                     # Current basin(NA, SA, EP, WP, SP, SI, NI)
wmo_wind = data.wmo_wind                # Maximum sustained wind speed(kts)
wmo_pres = data.wmo_pres                # Minimum central pressure(mb)


# Check the % of non-NaN data in the maximum sustained wind speed variable
for i in range(13664):
    not_NaN = 0
    values = data.wmo_wind[i].values
    for j in range(360):
        if values[j] == values[j]:    # NaN objects shapeshift, they are different from themselves
            not_NaN += 1
    not_NaN = not_NaN/360*100
    print(i, f"full of data at {not_NaN}%")


# Histogram plot of the amount of cyclones recordings per season
bins = np.unique(seasons)
counts, edges, bars = plt.hist(seasons, bins=bins, edgecolor='black')
plt.xlabel("Years", labelpad=12, fontsize=14)
plt.ylabel("Cyclones per year", labelpad=12, fontsize=14)
plt.xticks(bins, rotation=50)
plt.bar_label(bars)
plt.show()
