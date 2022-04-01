#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:42:28 2020

@author: walkerwhitfield
"""

import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


datadir = "/nexsan/people/rwhitfield/"
ds = xr.open_dataset(datadir + "modeldatarray.nc")
#print(ds)
maxwind = ds.maxwind
mean = maxwind.mean(dim='year')
std = maxwind.std(dim='year')

beta = np.sqrt(6)*(std/np.pi)
zeta = mean-(0.57721*beta)
ws = 75 #ENTER WINDSPEED VALUE IN MPH --> CONVERTED TO KTS
var = ((ws/1.151) - zeta)/beta #ENTER WINDSPEED VALUE IN MPH --> CONVERTED TO KTS

cdf = np.exp(((-1)*np.exp(-var)))

RP = (1/(1-cdf))
print(RP)



plt.figure(num=1, figsize=[10,5]) # start building the figure and specify the plot size
datacoord = ccrs.PlateCarree() # data coordinates: Use Plate Carree for data in lat-long coordinates
ax = plt.axes(projection=ccrs.PlateCarree())
lats = RP.latitude.values
lons = RP.longitude.values
max = 80
min = 0
interval = (max - min) / 250
color_levels = np.arange(min,max+interval,interval)
cs = plt.pcolor(lons,lats,RP.values,cmap='gnuplot2_r', vmin=min,vmax=max) #figure out why this isnt working
cbar = plt.colorbar(cs,shrink=0.70,pad=0.02) 
plt.title('Return Period (Years) of 75 mph Winds 1988-2017') # Title to show on plot
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

plt.savefig(datadir+ 'returnperiod.png', dpi=300, bbox_inches='tight')
