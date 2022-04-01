#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:32:34 2020

@author: walkerwhitfield
"""
import pandas as pd
import xarray as xr
import math
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import scipy.stats as stats
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P, Fit_Gamma_3P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Probability_plotting import plot_points
from reliability.Other_functions import make_right_censored_data, histogram

##################################################################
'''
datadir = "/Users/walkerwhitfield/Desktop/Research/"
dataset = pd.read_excel(datadir + "TestLocationsXLS.xls")
location = 'wsDB'
data = dataset[location].tolist()
data = [x for x in data if str(x) !='nan'] #removes nan values
#print(data)
'''
###############################################################
datadir = "/Users/walkerwhitfield/Desktop/Research/"
ds = xr.open_dataset(datadir + "maxwindTCs.nc")
point = ds.sel(latitude=25.74,longitude=-72.02, method = 'nearest')
maxwind = point['maxwind'].values

#print("maxwind values", maxwind)

data = maxwind[~np.isnan(maxwind)]
data = maxwind[maxwind !=0]

print(data)

####################################################################
'''
datadir = "/Users/walkerwhitfield/Desktop/Research/"
ds = xr.open_dataset(datadir + "maxwindTCs.nc")


val = ds.maxwind.sel(latitude=25,longitude=-75,method='nearest')
val = val.values
val = [i for i in val if i != 0] #removes zeros
print("Max non-zero TC wind speeds:",val)
data = val
'''
#################################################################


wbf = Fit_Weibull_3P(failures=data, show_probability_plot=False, print_results=False)  # fit the Weibull_3P distribution
print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
histogram(data,bins=6)
wbf.distribution.PDF(label='Weibull 3P Fit', linestyle='--')  # plots to PDF of the fitted Weibull_3P
gf = Fit_Gamma_3P(failures=data, show_probability_plot=False, print_results=False)
gf.distribution.PDF(label='Gamma 3P Fit')



mean = sum(data)/len(data)
std = np.std(data)
beta = np.sqrt(6)*(std/np.pi)
zeta = mean-(0.57721*beta)

winds = list(range(0,149)) #range of winds from which gumbel parameters are calculated for theoretical




#gumbel paramters are calculated at the point data and the theoretical distribution is plotted
#with the wind speed values specified above

gumbel = []
for rp in winds:
    z = (rp-zeta)/beta
    pdf = len(data)*np.exp(-(z+np.exp(-z)))/beta
    gumbel.append((pdf))
    #print(pdf)

#gumbel = ['%.3f' % elem for elem in gumbel]
divgum = sum(gumbel)
gumbelnorm = [y/divgum for y in gumbel] #normalizes the data so sum=1


plt.plot(winds,gumbelnorm, label = 'Gumbel Fit')





plt.legend()
plt.xlabel("Max Wind Speeds (mph)")
plt.ylabel("Fraction of Occurrence")
#plt.xlim(45,200)
#plt.title("Cross Correlation With Sigma=0.05")
plt.show()




loc = 0 #location of dataset to be plotted
mean = sum(data)/len(data)
var = np.var(data)
scale = 0.5*(mean-loc)*(((mean-loc)**2)/var+1)
shape = 0.5*(((mean-loc)**2)/var-1)
print("GDP PARAMETERS for : scale:",scale, ",", "shape:", shape, ",", "location:", loc)


winds = list(range(0,149)) #range of winds from which parameters are calculated for theoretical dist

#gumbel paramters are calculated at the point data and the theoretical distribution is plotted
#with the wind speed values specified above

gpd = []
for rp in winds:
    pdf = (scale**(-1))*(1-shape*(rp-loc)/scale)**(1/shape-1)
    gpd.append((pdf))
    
    cdf = 1-(1-shape*(rp-loc)/scale)**(1/shape)
    ret = 1/(1-cdf)
    
    #gpd.append((cdf))
    
    #print(rp, gpd)

#gumbel = ['%.3f' % elem for elem in gumbel]
gpd = [0 if math.isnan(x) else x for x in gpd] #changes nan in list to zeros
divgpd = sum(gpd)
gpdnorm = [y/divgpd for y in gpd] #normalizes the data so sum=1


#plt.hist(data, bins = 20, density=True)
plt.plot(winds, gpd, label = 'GPD Fit')
plt.legend()
plt.title("Fitted Distributions of HURDAT TC Winds for \n Location With n=20 Data Points from 1988-2017")
plt.show


#the following converts the histogram to a pdf and calculates rsquared of the
#distribution and the data values

'''
count,bins,_ = plt.hist(data,bins = 99, range=[50,149])

xcount = []
xbin = []
for n, b in zip(count,bins):
    xcount.append(n)
    xbin.append(b)
    
print(xbin, xcount)
'''


y,binedges = np.histogram(data, bins=149, range=[0,149], density = True) #bins and range must match windspeed range
bincenters = 0.5*(binedges[1:]+binedges[:-1])
#plt.plot(bincenters,y)

histpdf = y.tolist()
#plt.plot(winds,histpdf)



slope, intercept, r_value, p_value, std_err = stats.linregress(histpdf,gpd)
print("rsquared value for data and GPD is:",(r_value)**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(histpdf,gumbelnorm)
print("rsquared value for data and gumbel is:",(r_value)**2)