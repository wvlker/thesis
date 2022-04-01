import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

##########

#the following opens the datasets,
#the stochastic dataset is reformatted so the year in the header
#becomes another column for easier return period calculations

#it then adds the lat/lons to the corresponding rows of locids and saves
#the entire thing as one dataframe

datadir = "/nexsan/people/cocke/walker/"

storms = datadir + "20181002_m2b_stochastic_59000yrs_AT_1min_actualRough/"

df2 = pd.read_csv(datadir + "formm2_policies_22aug2016.csv", names = ['locid','gridid','lon','lat'])
df2 = df2.drop(columns=['gridid'])

files = sorted(glob.glob(storms + '/*.dat')) #runs files in name order


dfs = []

for file in files:
    df = pd.read_csv(file, sep='\n', header=None, dtype=str)
    df['id/date'] = df.iloc[0, 0]
    df['record#'] = df.iloc[1, 0]

    df['year'] = df['id/date'].str.extract(' (\d+)$')
    df[['locid', 'ws(mph)']] = df[0].str.split(',', n=1, expand=True).values

    cols = ['locid', 'ws(mph)', 'year','id/date', 'record#']
    dfn = df.dropna()[cols] #use this definition from now on for this df
    dfn['locid'] = dfn['locid'].astype(str).astype(int) #changes locid from object to int64
    dfn['ws(mph)'] = dfn['ws(mph)'].astype(float) #changes ws from object to float
    dfn = dfn.drop(columns=['id/date','record#'])
    
    lon_dict = df2.set_index('locid').to_dict()['lon']
    lat_dict = df2.set_index('locid').to_dict()['lat']

    dfn['newlons'] = dfn['locid'].map(lon_dict.get)
    dfn['newlats'] = dfn['locid'].map(lat_dict.get)

    print(file)
    
    
    dfs.append(dfn)
    
    
final_df = pd.concat(dfs,ignore_index=True)


print(final_df)

final_df.to_csv(r'/nexsan/people/rwhitfield/modeldata.csv')

