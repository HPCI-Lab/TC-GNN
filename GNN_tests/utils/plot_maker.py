import matplotlib.pyplot as plt
import xarray as xr

##### GRID OF ERA5 FEATURES #####
data = xr.open_dataset('./raw/year_1981_cyclone_18.nc')
fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(8, 16))
#fig.suptitle('Time: ' + str(data.time.values[0])[:23], fontsize=12, y=0.95)
row, col = 0, 0

keys = []
for key in data.data_vars:
    keys.append(key)

if 'Ymsk' in keys: keys.remove('Ymsk')

#axs[2, 0].axis('off')
axs[3, 1].axis('off')

for key in keys:
    ax = axs[row, col]
    pcm = ax.imshow(data.data_vars[key].values.squeeze(), extent=[data.data_vars[key].lon.values[0], data.data_vars[key].lon.values[-1], data.data_vars[key].lat.values[-1], data.data_vars[key].lat.values[0]])
    #if key=='Ymsk':
        #bar = fig.colorbar(pcm, location='top', orientation='horizontal', label='IBTrACS puncutal cyclone location')
    if key=='fg10':
        ax.title.set_text(key+' [m/s]')
        fig.colorbar(pcm, orientation='vertical', fraction=0.046, pad=0.04)#'10 metre wind gust since last post-process (6h) [m/s]')
    elif key=='i10fg':
        ax.title.set_text(key+' [m/s]')
        fig.colorbar(pcm, orientation='vertical', fraction=0.046, pad=0.04)#, 'Instantaneous 10 metre wind gust [m/s]')
    elif key=='vo_850':
        ax.title.set_text(key+' [1/s]')
        fig.colorbar(pcm, orientation='vertical', fraction=0.046, pad=0.04)#'Vorticity (relative) at 850 mb [1/s]', format='%.0e')
    elif key=='msl':
        ax.title.set_text(key+' [1/s]')
        fig.colorbar(pcm, orientation='vertical', fraction=0.046, pad=0.04)#'Vorticity (relative) at 850 mb [1/s]', format='%.0e')
    else:
        ax.title.set_text(key+' ['+data.data_vars[key].units+']')
        fig.colorbar(pcm, orientation='vertical', fraction=0.046, pad=0.04)#data.data_vars[key].long_name)
    if col==1:
        col = 0
        row += 1
    else:
        col += 1

#plt.ylabel("Latitude", fontsize=12)
#plt.xlabel("Longitude", fontsize=12)
plt.tight_layout(pad=5)
#plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.show()
