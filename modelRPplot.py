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

#this code saves the windspeeds of a specified return period in a netcdf 

datadir = "/nexsan/people/rwhitfield/"

print("Opening the data...")
ds = xr.open_dataset(datadir + "modeldataarrayHR.nc")

maxwind = ds['maxwind']
lat = ds['lat']
lon = ds['lon']


year = 100
wind_at_returnperiod = xr.DataArray(np.full((len(lat), len(lon)), np.nan), dims=['lat','lon'],coords=[lat,lon],name='maxwindatreturnperiod') #create new array with dims (time, lat, lon)
for i in range(0,len(lat)):
	print("solving at lat:", lat[i].values)
	for j in range(0, len(lon)):
		print("solving at lon:", lon[j].values)
		#get 1D list of max winds for all years in a given lat/lon cell, and sort them. 
		sorted1D = maxwind[:,i,j].sortby(maxwind[:,i,j], ascending=False).values 
		sorted1D = sorted1D[~np.isnan(sorted1D)] #remove nans, should be a shorter 1D array now
		exec = 1/year
		rank= 59000*exec 
		rankint = round(rank)
		if len(sorted1D) <= rankint:
			continue #if the grid cell doesnt have 590 values after  dropping NaNs, fill continue and NaN is left there
		else: #otherwise take the 590th index values of the  winds and place into new array at i,j location
			wind_at_returnperiod[i,j] = sorted1D[rankint]

wind_at_returnperiod.to_netcdf(datadir + f"RPfor{year}yearsHR.nc")









'''

year = 100 #return period for which you are calculating maxwind at a given point

sortedarray = np.full(maxwind.values.shape,np.nan) #creates new array with dims time, lat, lon
for i in range(0,len(lat)):
	for j in range(0,len(lon)):
		sortedarray[:,i,j] = maxwind[:,i,j].sortby(maxwind[:,i,j], ascending=False).values
		sortedarray = sortedarray[~np.isnan(maxwind)] #removes nans
		
		#below calculates return period from ranks of maxwinds
		exc = 1/year
		#print(year)
		#print(exc)
		rank = 59000*exc
		rankint = round(rank) #turns decimal rank from calculation to nearest whole
		#print(rankint)
		wind = sortedarray[rankint] #pulls wind at that given rank
		
		
		
print(sortedarray)







for i in lat
	for j in lon
	
	descend = np.sort('maxwind', ascending = False)

	maxwind = descend['maxwind'].values

	
	#np.set_printoptions(threshold=np.inf) #sets print options to full array vs truncated 

	#print(maxwind)

	year = 100 #this is the return period value in years for x wind speed


	exc = 1/year
	#print(year)
	#print(exc)
	rank = 59000*exc
	rankint = round(rank)
	#print(rankint)
	wind = maxwind[rankint]
	
	array2d[i,j] = 



#ws = maxwind.tolist() #pulls the value at that rank
#print (f"The return period for {ws} is", rp)


#below is other RP method

maxwind = ds.maxwind
mean = (maxwind.sum(dim='year'))/59000
#mean = maxwind.mean(dim='year')
std = maxwind.std(dim='year')

beta = np.sqrt(6)*(std/np.pi)
zeta = mean-(0.57721*beta)
ws = 75 #ENTER WINDSPEED VALUE IN MPH --> CONVERTED TO KTS
var = ((ws/1.151) - zeta)/beta #ENTER WINDSPEED VALUE IN MPH --> CONVERTED TO KTS

cdf = np.exp(((-1)*np.exp(-var)))

RP = (1/(1-cdf))
print(RP)
'''

'''
plt.figure(num=1, figsize=[10,5]) # start building the figure and specify the plot size
datacoord = ccrs.PlateCarree() # data coordinates: Use Plate Carree for data in lat-long coordinates
ax = plt.axes(projection=ccrs.PlateCarree())
lats = lat.values
lons = lon.values
#max = 80
#min = 0
#interval = (max - min) / 250
#color_levels = np.arange(min,max+interval,interval)
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
'''
