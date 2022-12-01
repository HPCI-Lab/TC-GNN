import geopandas as gpd  # geopandas for earth contours
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

data = xr.open_dataset('./data/IBTrACS/IBTrACS.since1980.v04r00.nc')
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the storms set on given latitude, longitude and data source
def plot_storms(these_storms, lon_cyclone=np.array([]), lat_cyclone=np.array([]), data_cyclone=np.array([]),
                                lon_mesh=np.array([]), lat_mesh=np.array([])):
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot every country
    #countries.plot(color='grey', ax=ax)
    # Plot single continents/countries
    #countries[countries["continent"] == "Asia"].plot(color='grey', ax=ax)
    #countries[countries["name"] == "Australia"].plot(color='grey', ax=ax)
    #st = 235
    if not lon_cyclone.any():                       # Process cyclone longitude
        lon_cyclone = data.lon
        print("Standard cyclone longitude in use.")
    else:
        print("Custom cyclone longitude in use.")
    if not lat_cyclone.any():                       # Process cyclone latitude
        print("Standard cyclone latitude in use.")
        lat_cyclone = data.lat
    else:
        print("Custom cyclone latitude in use.")
    if not data_cyclone.any():                      # Process cyclone data
        data_cyclone = data.wmo_wind
        print("Standard cyclone wmo_wind in use.")
    else:
        print("Custom cyclone data in use.")

    # Plot the mesh(if there is one)
    if not lon_mesh.any() and not lat_mesh.any():
        print("No mesh data was passed.")
    if lon_mesh.any() and lat_mesh.any():
        print("Custom mesh data in use.")
        plt.scatter(lon_mesh, lat_mesh, s=1)

    # Plot the cyclones
    for s in these_storms:
        plt.scatter(lon_cyclone[s].values, lat_cyclone[s].values, s=20, c=data_cyclone[s].values)    # TODO if a value is NaN, it's not gonna be plotted
    cbar = plt.colorbar(orientation='horizontal', pad=0.04)
    cbar.set_label(data_cyclone.long_name, labelpad=10)
    plt.show()

# Variables
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
data = data.drop_vars('usa_gust')       # maximum reported wind gust from a US agency
data = data.drop_vars('usa_seahgt')     # wave height for given radii(ft)
data = data.drop_vars('usa_searad')     # radial extent of given sea height(nmile)

# Tokyo-specific data
data = data.drop_vars('tokyo_lat')      # just lat, seems unnecessary
data = data.drop_vars('tokyo_lon')      # as above
data = data.drop_vars('tokyo_grade')    # storm grade from 2 to 9
data = data.drop_vars('tokyo_r50_dir')  # direction of maximum radius of 50 knots winds, all cardinal points combinations
data = data.drop_vars('tokyo_r50_long') # maximum radius of 50 knots winds
data = data.drop_vars('tokyo_r50_short')# minimum radius of 50 knots winds
data = data.drop_vars('tokyo_r30_dir')  # direction of maximum radius of 50 knots winds
data = data.drop_vars('tokyo_r30_long') # maximum radius of 30 knots winds
data = data.drop_vars('tokyo_r30_short')# minimum radius of 30 knots winds
data = data.drop_vars('tokyo_land')     # landfall flag(0 or 1)

# Chinese-specific data
data = data.drop_vars('cma_lat')
data = data.drop_vars('cma_lon')
data = data.drop_vars('cma_cat')        # storm category(from 0 to 9)

# Hong-Kong-specific data
data = data.drop_vars('hko_lat')
data = data.drop_vars('hko_lon')
data = data.drop_vars('hko_cat')        # storm category(letters permutation between S, T, D)

# New Delhi-specific data
data = data.drop_vars('newdelhi_lat')
data = data.drop_vars('newdelhi_lon')
data = data.drop_vars('newdelhi_grade') # storm grade(letters permutation between D, C, S, E, V)
data = data.drop_vars('newdelhi_ci')    # Dvorak current intensity(RWMC New Delhi)(between 0.0 and 8.0)
data = data.drop_vars('newdelhi_dp')    # storm pressure drop
data = data.drop_vars('newdelhi_poci')  # pressure of outermost closed isobar

# RSMC La Reunion-specific data(french monitoring center of the South-West Indian Ocean)
data = data.drop_vars('reunion_lat')
data = data.drop_vars('reunion_lon')
data = data.drop_vars('reunion_type')   # storm type(from 1 to 9)
data = data.drop_vars('reunion_tnum')   # Dvorak t number(from 0.0 to 8.0)
data = data.drop_vars('reunion_ci')     # Dvorak Current Intensity (CI)(from 0.0 to 8.0)
data = data.drop_vars('reunion_rmw')    # radius of maximum winds
data = data.drop_vars('reunion_r34')    # radius of 34 knot winds(storm force)
data = data.drop_vars('reunion_r50')    # radius of 50 knot winds(gale force)
data = data.drop_vars('reunion_r64')    # radius of 64 knot winds(hurricane force)
data = data.drop_vars('reunion_gust')   # maximum reported wind gust from Reunion
data = data.drop_vars('reunion_gust_per')# time period of the wind gust from Reunion

# Australian BoM-specific data
data = data.drop_vars('bom_lat')
data = data.drop_vars('bom_lon')
data = data.drop_vars('bom_type')       # cyclone type(from 10 to 91)
data = data.drop_vars('bom_tnum')       # Dvorak t number(from 0.0 to 8.0)
data = data.drop_vars('bom_ci')         # Dvorak Current Intensity (CI)(from 0.0 to 8.0)
data = data.drop_vars('bom_rmw')        # radius of maximum winds
data = data.drop_vars('bom_r34')        # radius of 34 knot winds(storm force)
data = data.drop_vars('bom_r50')        # radius of 50 knot winds(gale force)
data = data.drop_vars('bom_r64')        # radius of 64 knot winds(hurricane force)
data = data.drop_vars('bom_roci')       # radius of outermost closed isobar
data = data.drop_vars('bom_poci')       # environmental pressure
data = data.drop_vars('bom_eye')        # eye diameter
data = data.drop_vars('bom_pos_method') # method used to derive position(from 1 to 13)
data = data.drop_vars('bom_pres_method')# method used to derive intensity(from 1 to 9)
data = data.drop_vars('bom_gust')       # maximum reported wind gust from BoM
data = data.drop_vars('bom_gust_per')   # time period of the wind gust from BoM

# RSMC Fiji-specific data
data = data.drop_vars('nadi_lat')
data = data.drop_vars('nadi_lon')
data = data.drop_vars('nadi_cat')       # storm category

# RSMC Wellington-specific data(New Zeland center)
data = data.drop_vars('wellington_lat')
data = data.drop_vars('wellington_lon')

# DS824 dataset-specific data
data = data.drop_vars('ds824_lat')
data = data.drop_vars('ds824_lon')
data = data.drop_vars('ds824_stage')    # storm classification

# TD-9636 dataset-specific data
data = data.drop_vars('td9636_lat')
data = data.drop_vars('td9636_lon')
data = data.drop_vars('td9636_stage')   # storm classification(from 0 to 7)

# TD-9635 dataset-specific data
data = data.drop_vars('td9635_lat')
data = data.drop_vars('td9635_lon')
data = data.drop_vars('td9635_roci')    # radius of outermost closed isobar

# Neumann dataset-specific data
data = data.drop_vars('neumann_lat')
data = data.drop_vars('neumann_lon')
data = data.drop_vars('neumann_class')  # storm classification

# M.L. Chenoweth dataset-specific data
data = data.drop_vars('mlc_lat')
data = data.drop_vars('mlc_lon')
data = data.drop_vars('mlc_class')      # storm classification

# Useful variables(?)
data = data.drop_vars('basin')          # Current basin(NA, SA, EP, WP, SP, SI, NI)
data = data.drop_vars('wmo_wind')       # Maximum sustained wind speed(kts) from Official WMO Agency
data = data.drop_vars('wmo_pres')       # Minimum central pressure(mb) from Official WMO Agency
data = data.drop_vars('track_type')     # name of track type(MAIN or SPUR, read the description)
data = data.drop_vars('main_track_sid') # as above, check the description
data = data.drop_vars('storm_speed')    # Storm translation speed(kts)
data = data.drop_vars('storm_dir')      # Storm translation direction(degrees)
data = data.drop_vars('usa_wind')       # Maximum sustained wind speed  from USA, for some storms it contains better data
data = data.drop_vars('usa_pres')       # Minimum central pressure      from USA, for some storms it contains better data
data = data.drop_vars('tokyo_wind')
data = data.drop_vars('tokyo_pres')
data = data.drop_vars('cma_wind')
data = data.drop_vars('cma_pres')
data = data.drop_vars('hko_wind')
data = data.drop_vars('hko_pres')
data = data.drop_vars('newdelhi_wind')
data = data.drop_vars('newdelhi_pres')
data = data.drop_vars('reunion_wind')
data = data.drop_vars('reunion_pres')
data = data.drop_vars('bom_wind')
data = data.drop_vars('bom_pres')
data = data.drop_vars('nadi_wind')
data = data.drop_vars('nadi_pres')
data = data.drop_vars('wellington_wind')
data = data.drop_vars('wellington_pres')
data = data.drop_vars('ds824_wind')
data = data.drop_vars('ds824_pres')
data = data.drop_vars('td9636_wind')
data = data.drop_vars('td9636_pres')
data = data.drop_vars('td9635_wind')
data = data.drop_vars('td9635_pres')
data = data.drop_vars('neumann_wind')
data = data.drop_vars('neumann_pres')
data = data.drop_vars('mlc_wind')
data = data.drop_vars('mlc_pres')


# Check the % of non-NaN data in the passed variable, and find the highest
# Be careful: NaN data is perfectly fine, since we have 360 slots(one per day of the year) and the storms are active just on a subset of these
# This may be useful to compare WMO and US results and pick the one with the highest value coverage for a particular storm
def find_NaN(data_var):      # es. find_NaN(data.wmo_wind)
    variable = data_var
    highest = [0, 0]
    for s in range(data.storm.size):
        not_NaN = 0
        values = variable[s].values
        for t in range(data.date_time.size):
            if values[t] == values[t]:    # NaN objects shapeshift, they are different from themselves. Can also do != b'' for other variables
                not_NaN += 1
        not_NaN = not_NaN/data.date_time.size*100
        if highest[1] < not_NaN:
            highest[0] = s
            highest[1] = not_NaN
        print(s, f"data full at {not_NaN}%")
    print('[', highest[0], highest[1], ']')

# Counts the overall storm observations in each basin
def count_storms():
    basins = [b'EP', b'WP', b'SP', b'NA', b'SA', b'NI', b'SI', b'']
    tmp = data.basin.values
    basins_count = [0]*len(basins)
    for s in range(data.storm.size):
        for t in range(data.date_time.size):
            for b in range(len(basins)):
                if tmp[s][t] == basins[b]:
                    basins_count[b] += 1
    print(basins, "\n", basins_count)

# Finds storm ids for storms belonging to some basin(in byte format, e.g. b'WP')
def extract_basin(this_basin):
    storms = []
    tmp = data.basin.values
    for s in range(data.storm.size):
        for t in range(data.date_time.size):
            if tmp[s][t] == this_basin:
                if s not in storms:
                    storms.append(s)
    print(f"Found {len(storms)} storms crossing at least once the basin {this_basin}")
    return storms

# Calculates percentage of a basin occurrencies in a given set of storms
def rates_of_basins(this_basin, these_storms):
    basin_rates = []
    tmp = data.basin.values
    for s in these_storms:
        this_basin_rate = 0
        for t in range(data.date_time.size):
            if tmp[s][t] == this_basin:
                this_basin_rate += 1
        this_basin_rate = this_basin_rate/data.date_time.size*100
        basin_rates.append(this_basin_rate)
    print(f"Rates of basin {this_basin} calculated.")
    return basin_rates

# Calculates the most extreme points where these storms were recorded
def boundaries_of_storms(these_storms, lon_source=np.array([]), lat_source=np.array([])):
    if not lat_source.any():
        lat = data.lat.values
        print("Standard latitude in use.")
    else:
        lat = lat_source.values
        print("Custom latitude in use.")
    if not lon_source.any():
        lon = data.lon.values
        print("Standard longitude in use.")
    else:
        lon = lon_source.values
        print("Custom longitude in use.")
    # Opposite values are set here to favor their correction
    left = 180
    right = -180
    bottom = 90
    top = -90
    for s in these_storms:
        for t in range(data.date_time.size):
            if lon[s][t] > right:
                right = lon[s][t]
            if lon[s][t] < left:
                left = lon[s][t]
            if lat[s][t] < bottom:
                bottom = lat[s][t]
            if lat[s][t] > top:
                top = lat[s][t]
    print(f"Boundaries of storms:\nleft: {left}\nright: {right}\nbottom: {bottom}\ntop: {top}")

# Histogram plot of the amount of cyclones recordings per season
seasons = data.season.values
bins = np.unique(seasons)
counts, edges, bars = plt.hist(seasons, bins=bins, edgecolor='black')
plt.xlabel("Years", labelpad=12, fontsize=14)
plt.ylabel("Cyclones per year", labelpad=12, fontsize=14)
plt.xticks(bins, rotation=50)
plt.bar_label(bars)
plt.show()
