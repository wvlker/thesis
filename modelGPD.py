#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

###########
#this code produces the GPD pdf distribution plots

datadir = "/nexsan/people/rwhitfield/"
ds = xr.open_dataset(datadir + "modeldataarray.nc")

point = ds.sel(lat=27.722967,lon=-82.655764, method = 'nearest')

maxwind = point['maxwind'].values

maxwind = maxwind[~np.isnan(maxwind)] #removes nans


loc = 50 #plotting data fit starting at 50
mean = maxwind.mean()
print("mean is: ", mean)
var = np.var(maxwind)
print("var is:",var)
scale = 0.5*(mean-loc)*(((mean-loc)**2)/var+1)
shape = 0.5*(((mean-loc)**2)/var-1)
print("scale is:",scale, "and", "shape is:", shape)


winds = list(range(50,149)) #range of winds from which gumbel parameters are calculated for theoretical
#note high wind values will induce a value error for the pdf calculation because of very low tail values, still plots though

#gumbel paramters are calculated at the point data and the theoretical distribution is plotted
#with the wind speed values specified above

gpd = []
for mph in winds:
    pdf = (scale**(-1))*(1-shape*(mph-loc)/scale)**(1/shape-1) #this line with induce value error at high mph values
    gpd.append((pdf))
    #print(rp, gpd)

#gumbel = ['%.3f' % elem for elem in gumbel]
divgpd = sum(gpd)
gpdnorm = [y/divgpd for y in gpd] #normalizes the data so sum=1


#plt.hist(data, bins = 20, density=True)
plt.plot(winds, gpd, label = 'GPD Fit')


ax = sns.distplot(maxwind, bins = 15)


plt.title("Probability Density Function for the Generalized Pareto Distribution of \n Extreme Hurricane Winds in Tampa, FL: ")
plt.xlabel("Windspeeds(mph)")
plt.ylabel("Probability Density")
plt.savefig(datadir+'Tampa.png',dpi=300, bbox_inches='tight')

