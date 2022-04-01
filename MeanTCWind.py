import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

datadir = "/Users/walkerwhitfield/Desktop/Research/"
ds = xr.open_dataset(datadir + "maxwindTCs.nc")
maxwind = ds.maxwind
mean = maxwind.mean(dim='year')


plt.figure(num=1, figsize=[10,5]) # start building the figure and specify the plot size
datacoord = ccrs.PlateCarree() # data coordinates: Use Plate Carree for data in lat-long coordinates
ax = plt.axes(projection=ccrs.PlateCarree())
lats = mean.latitude.values
lons = mean.longitude.values
max = 80
min = 0
interval = (max - min) / 250
color_levels = np.arange(min,max+interval,interval)
cs = plt.pcolormesh(lons,lats,mean.values,cmap='gnuplot2_r',vmin=min,vmax=max)
cbar = plt.colorbar(cs,shrink=0.70,pad=0.02) 
plt.title('Mean TC Winds (km/h) 1988-2017') # Title to show on plot
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylabel_style = {'size': 7}
gl.xlabel_style = {'size': 7}
ax.coastlines(resolution='50m') # Add coasts: Use '110m' or '50m' for closer resolution
ax.add_feature(cfeature.STATES.with_scale('50m')) # Add states as above
