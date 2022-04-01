import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

datadir = "/nexsan/people/rwhitfield/"

datadir2 = "/nexsan/people/cocke/walker/"

print("opening the data!!!!")
df = pd.read_csv(datadir + "modeldata.csv", header = 0)


print('narrowing dataframe to Tampa region...')
df = df[ (df['newlats']>27) & (df['newlats']<28.5) & (df['newlons']>-83) & (df['newlons']<-82)]




ulats = np.arange(27.6, 28, .1/111)
ulons = np.arange(-82.9, -82.6, .1/111)

#finding 
f = interp1d(ulats,np.arange(ulats.size),kind='nearest',fill_value='extrapolate')
idx_lat = f(df.newlats.values).astype(np.int)

#fancylons = np.mod( df.nee)	
g = interp1d(ulons,np.arange(ulons.size),kind='nearest',fill_value='extrapolate')
idx_lon = g(df.newlons.values).astype(np.int)

print('adding regrid index\n',df)
uindex = idx_lat * 1000 + idx_lon
df['idx_lat'] = idx_lat
df['idx_lon'] = idx_lon
df['regrid_index'] = uindex



print("grouping data by year/gridpoint !!!!!!")
groupdf = df.groupby(['year','regrid_index', 'idx_lat', 'idx_lon'],as_index=False).max() #separates df into years, saving max ws value per lat/lon

print('df grouped by year\n',groupdf)

#ds = pd.read_csv(datadir2 + "formm2_policies_22aug2016.csv", names = ['locid', ' gridid', 'lons','lats'])
uyear = np.unique(groupdf.year.values)
print('number of years with storm in Tampa area: ', len(uyear))




allyears=[]

for year in uyear:

	array2d = xr.DataArray(np.full((len(ulats),len(ulons)),np.nan),dims=['lat','lon'],coords=[ulats,ulons],name='maxwind') #dataarray changes np to xr 
	
	dfbyyear = groupdf[groupdf['year']==year] #pulls year from groupdf
	maxwindforyear = dfbyyear['ws(mph)'].values
	
	print(year)
	
	
	for i,j,maxwind in zip(dfbyyear.idx_lat.values, dfbyyear.idx_lon.values, maxwindforyear):
		
		array2d[i,j] = maxwind
		#array2d.loc[dict(lat=lat,lon=lon)]=maxwind #loop through pd dataframe/year, lat/lon maxwind in new 2d array2d
		
	#array2d.to_netcdf(datadir + f"griddedmodelyear/gridmodel_{year}.nc")


	allyears.append(array2d)

allyears=xr.concat(allyears,dim='year').assign_coords(year=uyear)

allyears.to_netcdf(datadir + "modeldataarrayHR.nc")

print("all done!")


