#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:18:46 2019

@author: walkerwhitfield
"""

import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def haversine(centerLat, centerLon, lat2, lon2):  
    # distance between latitudes 
    # and longitudes 
    dLat = (lat2 - centerLat) * np.pi / 180.0
    dLon = (lon2 - centerLon) * np.pi / 180.0
    # convert to radians 
    centerLat = (centerLat) * np.pi / 180.0
    lat2 = (lat2) * np.pi / 180.0
    # apply formulae 
    a = (pow(np.sin(dLat / 2), 2) + 
            pow(np.sin(dLon / 2), 2) * 
                np.cos(centerLat) * np.cos(lat2)) 
    rad = 6371
    c = 2 * np.arcsin(np.sqrt(a)) 
    dist = rad * c 
    return dist




datadir = "/Users/walkerwhitfield/Desktop/Research/"
df = pd.read_csv(datadir + "HURDAT2_ORG.csv", names=['ID#', 'NAME', 'MONTH/DAY/HOUR','YEAR' , 'LAT(N)', 'LON(W)', 'MAXWIND(kt)', 'MINP(hPa)', 'MAXWINDR(nm)', 'EYEDIAM(nm)', 'OUTERISOP(hPa)', 'OUTERISOR(nm)', '34ktNE', '34ktSE', '34ktSW', '34ktNW', '50ktNE', '50ktSE' , '50ktSW', '50ktNW', '64ktNE', '64ktSE', '64ktSW', '64ktNW', 'STORMTYPE','NMFROMLAND'])
df = df.drop(columns=['MINP(hPa)','EYEDIAM(nm)', 'OUTERISOP(hPa)', 'OUTERISOR(nm)','STORMTYPE','NMFROMLAND'])
ds = xr.open_dataset(datadir + "ref_grid.nc")
ds = ds.sortby('latitude',ascending=True)
example_grid = xr.full_like(ds.sst,0)  #defines grid dimensions by variable and populates with 0
example_grid.name = 'maxwind'



groupdf = df.groupby(['YEAR'])

#del ds

final_maingrid = []
year_list = []

for year in range(1988,2017+1):
    
    maingrid = xr.full_like(ds.sst,0)  #maingrid changes each year (not each time step)
    maingrid.name = 'maxwind'
    maingrid = maingrid.drop('time')
    maingrid.attrs['long_name'] = 'Max Wind from a TC in this Year'
    maingrid.attrs['units'] = 'year'
    
    dfbyyear = groupdf.get_group(year)
    
    for t in range(0,len(dfbyyear)):  #run through given length of variable list
        
        row = dfbyyear.iloc[t]  #iloc (index location) dictates row at each position
        
        tc_lat = row['LAT(N)']
        tc_lon = row['LON(W)'] * -1
        maxwind = row['MAXWIND(kt)']
        maxwindr = row['MAXWINDR(nm)'] * 1.852 #convert to km
        wind_34ktNE = row['34ktNE'] * 1.852 #WIND SPEED IS NOW IN KPH
        wind_34ktSE = row['34ktSE'] * 1.852 
        wind_34ktSW = row['34ktSW'] * 1.852
        wind_34ktNW = row['34ktNW'] * 1.852
        wind_50ktNE = row['50ktNE'] * 1.852
        wind_50ktSE = row['50ktSE'] * 1.852
        wind_50ktSW = row['50ktSW'] * 1.852
        wind_50ktNW = row['50ktNW'] * 1.852
        wind_64ktNE = row['64ktNE'] * 1.852
        wind_64ktSE = row['64ktSE'] * 1.852
        wind_64ktSW = row['64ktSW'] * 1.852
        wind_64ktNW = row['64ktNW'] * 1.852
        

        #defining subgrid quadrants to be placed on the maingrid
        subgrid_NE = example_grid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon,tc_lon+7.5))
        lon2d, lat2d = np.meshgrid(subgrid_NE.longitude.values, subgrid_NE.latitude.values)
        dist_NE = haversine(tc_lat,tc_lon,lat2d,lon2d)
        #dist_NE = xr.DataArray(dist_NE,coords)

        subgrid_NW = example_grid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon-7.5,tc_lon))
        lon2d, lat2d = np.meshgrid(subgrid_NW.longitude.values, subgrid_NW.latitude.values)
        dist_NW = haversine(tc_lat,tc_lon,lat2d,lon2d)   
        
        subgrid_SW = example_grid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon-7.5,tc_lon))
        lon2d, lat2d = np.meshgrid(subgrid_SW.longitude.values, subgrid_SW.latitude.values)
        dist_SW = haversine(tc_lat,tc_lon,lat2d,lon2d)
        
        subgrid_SE = example_grid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon,tc_lon+7.5))
        lon2d, lat2d = np.meshgrid(subgrid_SE.longitude.values, subgrid_SE.latitude.values)
        dist_SE = haversine(tc_lat,tc_lon,lat2d,lon2d)
        

        #grid = example_grid
        #defining subgrid thresholds to replace grid cells with largest value
        #WINDSPEED CONVERTED BACK TO KTS FOR ANALYSIS
        grid_NE = np.zeros(dist_NE.shape)
        if wind_34ktNE/1.852 != -99:      #WIND SPEED CONVERTED BACK TO KT FOR ANALYSIS
            grid_NE = np.where(wind_34ktNE - dist_NE >= 0, 34, 0)
        if wind_50ktNE/1.852 != -99:
            grid_NE = np.where(wind_50ktNE - dist_NE >= 0, 50, grid_NE)
        if wind_64ktNE/1.852 != -99:
            grid_NE = np.where(wind_64ktNE - dist_NE >= 0, 64, grid_NE)
        if (maxwindr/1.852) != -99 or (maxwind/1.852) != -99:
            grid_NE = np.where(maxwindr - dist_NE >= 0, maxwind, grid_NE)
        
        
        grid_NW = np.zeros(dist_NW.shape)
        if wind_34ktNW/1.852 != -99:
            grid_NW = np.where(wind_34ktNW - dist_NW >= 0, 34, 0)
        if wind_50ktNW/1.852 != -99:
            grid_NW = np.where(wind_50ktNW - dist_NW >= 0, 50, grid_NW)
        if wind_64ktNW/1.852 != -99:
            grid_NW = np.where(wind_64ktNW - dist_NW >= 0, 64, grid_NW)
        if (maxwindr/1.852) != -99 or (maxwind/1.852) != -99:
            grid_NW = np.where(maxwindr - dist_NW >= 0, maxwind, grid_NW)
        
        
        grid_SW = np.zeros(dist_SW.shape)
        if wind_34ktSW/1.852 != -99:
            grid_SW = np.where(wind_34ktSW - dist_SW >= 0, 34, 0)
        if wind_50ktSW/1.852 != -99:
            grid_SW = np.where(wind_50ktSW - dist_SW >= 0, 50, grid_SW)
        if wind_64ktSW/1.852 != -99:
            grid_SW = np.where(wind_64ktSW - dist_SW >= 0, 64, grid_SW)
        if (maxwindr/1.852) != -99 or (maxwind/1.852) != -99:
            grid_SW = np.where(maxwindr - dist_SW >= 0, maxwind, grid_SW)
        
        
        grid_SE = np.zeros(dist_SE.shape)
        if wind_34ktSE/1.852 != -99:
            grid_SE = np.where(wind_34ktSE - dist_SE >= 0, 34, 0)
        if wind_50ktSE/1.852 != -99:
            grid_SE = np.where(wind_50ktSE - dist_SE >= 0, 50, grid_SE)
        if wind_64ktSE/1.852 != -99:
            grid_SE = np.where(wind_64ktSE - dist_SE >= 0, 64, grid_SE)
        if (maxwindr/1.852) != -99 or (maxwind/1.852) != -99:
            grid_SE = np.where(maxwindr - dist_SE >= 0, maxwind, grid_SE)
        
        
        # NE ---- Filling in the main grid with the NE quadrant at this time step
        #.loc changes grid cells within larger domain based on values within subgrid domains, specifying grid domain change area each time, cannot be arbitrary
        maingrid.loc[dict(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon,tc_lon+7.5))] = \
                np.where(grid_NE > maingrid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon,tc_lon+7.5)),\
                         grid_NE, maingrid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon,tc_lon+7.5)))
        
        maingrid.loc[dict(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon-7.5,tc_lon))] = \
                np.where(grid_NW > maingrid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon-7.5,tc_lon)),\
                         grid_NW, maingrid.sel(latitude=slice(tc_lat,tc_lat+7.5),longitude=slice(tc_lon-7.5,tc_lon)))
                
        maingrid.loc[dict(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon-7.5,tc_lon))] = \
                np.where(grid_SW > maingrid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon-7.5,tc_lon)),\
                         grid_SW, maingrid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon-7.5,tc_lon)))
                
        maingrid.loc[dict(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon,tc_lon+7.5))] = \
                np.where(grid_SE > maingrid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon,tc_lon+7.5)),\
                         grid_SE, maingrid.sel(latitude=slice(tc_lat-7.5,tc_lat),longitude=slice(tc_lon,tc_lon+7.5)))

        print('Year: ',year,' t: ',t,' of ',len(dfbyyear)) #shows what time step put of total it is at, i.e. how long it's taking
      
        
    maingrid.to_netcdf(datadir+f'OutputData/MaxWind_{year}.nc') #add f before string when adding changing variable name 
    
    final_maingrid.append(maingrid)
    year_list.append(year)
    
    
    
    plt.figure(num=1, figsize=[10,5]) # start building the figure and specify the plot size
    datacoord = ccrs.PlateCarree() # data coordinates: Use Plate Carree for data in lat-long coordinates
    ax = plt.axes(projection=ccrs.PlateCarree()) # Map projection: PlateCarree for Global; LambertConformal for CONUS
    ax.set_extent([maingrid.longitude.min(), maingrid.longitude.max(), maingrid.latitude.min(), maingrid.latitude.max()], ccrs.PlateCarree()) # Set the extent of the map based on lat and long
    max = 155
    min = 0
    lats = maingrid.latitude.values
    lons = maingrid.longitude.values
    interval = (max - min) / 250.  #length of intervals determined by dividing by 150 bc I want 150 intervals
    color_levels = np.arange(min,max+interval,interval)
    #cs = plt.contourf(lons,lats,maingrid.values, levels = color_levels,cmap='gnuplot2_r',extend = 'max',vmin=0,vmax=0.15)
    cs = plt.pcolormesh(lons,lats,maingrid.values,cmap='gnuplot2_r',vmin=min,vmax=max)
    cbar = plt.colorbar(cs,shrink=0.70,pad=0.02) 
    plt.title(f'Max TC Winds in {year}') # Title to show on plot
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
    plt.show()
    plt.savefig(datadir+f'OutputPlots/MaxWind_{year}.png',dpi=300, bbox_inches='tight')
    plt.close()
    
    
    del maingrid
    
ds.close()
final_maingrid = xr.concat(final_maingrid, dim='year')
final_maingrid = final_maingrid.assign_coords(year=np.array(year_list))
final_maingrid.to_netcdf(datadir+'maxwindTCs.nc')
        
        
        
        
        #print(maxwind)
        

    #print(year)