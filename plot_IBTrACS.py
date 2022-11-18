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
#countries[countries["continent"] == "Asia"].plot(color='grey', ax=ax)
countries[countries["name"] == "Australia"].plot(color='grey', ax=ax)

st = 235
data_sample = data.wmo_wind #pres
plt.scatter(data.lon[st].values, data.lat[st].values, s=20, c=data_sample[st].values)    # TODO if a value is NaN, it's not gonna be plotted
cbar = plt.colorbar(orientation='horizontal', pad=0.04)
cbar.set_label(data_sample.long_name, labelpad=10)
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
data = data.drop_vars('wmo_agency')     # official WMO agency
data = data.drop_vars('dist2land')      # distance to the nearest land point
data = data.drop_vars('landfall')       # minimum distance to land between current location and next
data = data.drop_vars('iflag')          # source agencies

# USA-specific data
data = data.drop_vars('usa_agency')     # source dataset for U.S. data
data = data.drop_vars('usa_atcf_id')    # just an identifier
data = data.drop_vars('usa_lat')        # just lat, seems unnecessary
data = data.drop_vars('usa_lon')        # as above
data = data.drop_vars('usa_record')     # storm record type, check the description
data = data.drop_vars('usa_status')     # storm status, check description
data = data.drop_vars('usa_sshs')       # Saffir-Simpson Hurricane Wind Scale Category(from -5 to 5). Some storms don't have maximum wind but have values in here
data = data.drop_vars('usa_r34')        # radius of 34 knot winds
data = data.drop_vars('usa_r50')        # radius of 50 knot winds
data = data.drop_vars('usa_r64')        # radius of 64 knot winds
data = data.drop_vars('usa_poci')       # pressure of outermost closed isobar(not best tracked). Doesn't follow the usa_pres trends
data = data.drop_vars('usa_roci')       # as usa_poci, but the valid_min and valid_max are different. As above, doesn't follow the usa_pres trends
data = data.drop_vars('usa_rmw')        # radius of maximum winds (not best tracked)
data = data.drop_vars('usa_eye')        # eye diameter (not best tracked)

# Tokyo-specific data
data = data.drop_vars('tokyo_lat')      # just lat, seems unnecessary
data = data.drop_vars('tokyo_lon')      # as above
data = data.drop_vars('tokyo_grade')    # storm grade from 2 to 9
data = data.drop_vars('')
data = data.drop_vars('')

# Useful variables(?)
basins = data.basin                     # Current basin(NA, SA, EP, WP, SP, SI, NI)
wmo_wind = data.wmo_wind                # Maximum sustained wind speed(kts) from Official WMO Agency
wmo_pres = data.wmo_pres                # Minimum central pressure(mb) from Official WMO Agency
data = data.drop_vars('track_type')     # name of track type(MAIN or SPUR, read the description)
data = data.drop_vars('main_track_sid') # as above, check the description
data = data.drop_vars('usa_wind')       # Maximum sustained wind speed  from USA, for some storms it contains better data
data = data.drop_vars('usa_pres')       # Minimum central pressure      from USA, for some storms it contains better data
data = data.drop_vars('tokyo_wind')     # Maximum sustained wind speed  from Tokyo, for some storms it contains better data
data = data.drop_vars('tokyo_pres')     # Minimum central pressure      from Tokyo, for some storms it contains better data


# Check the % of non-NaN data in the passed variable, and find the highest
# Be careful: NaN data is perfectly fine, since we have 360 slots(one per day of the year) and the storms are active just on a subset of these
# This may be useful to compare WMO and US results and pick the one with the highest value coverage for a particular storm
def find_NaN(data_var):      # es. find_NaN(data.wmo_wind)
    variable = data_var
    highest = [0, 0]
    for i in range(data.storm.size):
        not_NaN = 0
        values = variable[i].values
        for j in range(data.date_time.size):
            if values[j] == values[j]:    # NaN objects shapeshift, they are different from themselves
                not_NaN += 1
        not_NaN = not_NaN/360*100
        if highest[1] < not_NaN:
            highest[0] = i
            highest[1] = not_NaN
        print(i, f"data full at {not_NaN}%")
    print('[', highest[0], highest[1], ']')


# Histogram plot of the amount of cyclones recordings per season
seasons = data.season.values
bins = np.unique(seasons)
counts, edges, bars = plt.hist(seasons, bins=bins, edgecolor='black')
plt.xlabel("Years", labelpad=12, fontsize=14)
plt.ylabel("Cyclones per year", labelpad=12, fontsize=14)
plt.xticks(bins, rotation=50)
plt.bar_label(bars)
plt.show()
