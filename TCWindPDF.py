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


datadir = "/Users/walkerwhitfield/Desktop/Research/"
ds = xr.open_dataset(datadir + "modeldataarray.nc")
#print(ds)
data = ds.maxwind

year = ds['year']

maxwindyr = data.isel(year=50)


print(data)
'''
################
# calculates with GDP distribution

#mean = data.mean(dim='year')

#print(mean)

loc = 1 #location of dataset to be plotted
mean = sum(data)/len(data)
var = np.var(data)
scale = 0.5*(mean-loc)*(((mean-loc)**2)/var+1)
shape = 0.5*(((mean-loc)**2)/var-1)


#gumbel paramters are calculated at the point data and the theoretical distribution is plotted
#with the wind speed values specified above

mph = 75

ws = mph/1.151 #converts ws from mph to kts, which the data is in kts    
cdf = 1-(1-shape*(ws-loc)/scale)**(1/shape)
ret = 1/(1-cdf)

'''
#################
#calculates RPs through gumbel distribution
'''
datadir = "/Users/walkerwhitfield/Desktop/Research/"
ds = xr.open_dataset(datadir + "maxwindTCs.nc")
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
'''


plt.figure(num=1, figsize=[10,5]) # start building the figure and specify the plot size
datacoord = ccrs.PlateCarree() # data coordinates: Use Plate Carree for data in lat-long coordinates
ax = plt.axes(projection=ccrs.PlateCarree())
lats = data.lat.values
lons = data.lon.values
max = 100
min = 50
interval = (max - min) / 250
color_levels = np.arange(min,max+interval,interval)
cs = plt.pcolor(lons,lats,maxwindyr.values,cmap='jet', vmin=min,vmax=max) #figure out why this isnt working
cbar = plt.colorbar(cs,shrink=0.70,pad=0.02) 
#cbar.set_ticks([0,20,40,60,80,100])
#cbar.set_ticklabels(['0','20','40','60','80','100+'])
#cbar.set_label('Years', rotation=270)
plt.title('Modeled FPLHM TC Winds (mph)') # Title to show on plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylabel_style = {'size': 7}
gl.xlabel_style = {'size': 7, 'rotation': 45}
ax.coastlines(resolution='10m') # Add coasts: Use '110m' or '50m' for closer resolution
ax.add_feature(cfeature.STATES.with_scale('10m')) # Add states as above


#plt.savefig(datadir+ 'RPhurdat.png', dpi=300, bbox_inches='tight')