#!/usr/bin/env python

import numpy as np
import matplotlib
from pylab import rcParams
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects.numpy2ri as rpyn

font = {'family' : 'CMU Sans Serif'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 25,
          'grid.linewidth': 0.2,
          'font.size': 25,
          'legend.fontsize': 20,
          'legend.frameon': False,
          'xtick.labelsize': 20,
          'xtick.direction': 'out',
          'ytick.labelsize': 20,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'text.usetex': False}
rcParams.update(params)

### Get the station ID
#data_dir = '../HCDN/EastHghlnds'
data_dir = '../HCDN/NEUS'
fig_dir = '/home/wind/hexg/Dropbox/Research_Princeton/GeneralExam/Results'
clidir = '/home/wind/hexg/Research/Data/Data_CEE599/Data_Dropbox/Data/ClimateIndices'
minLen = 60
#gageInfo = pd.read_excel('%s/HCDN-2009_Station_Info_EastHghlnds.xlsx' % (data_dir))
gageInfo = pd.read_excel('%s/HCDN-2009_Station_Info_NEUS.xlsx' % (data_dir))
gageID = gageInfo['STATION ID']
gageStime = gageInfo['stime (WY)']
gageDrainage = gageInfo['DRAIN_SQKM']
gageLat = gageInfo['LAT_GAGE']
gageLon = gageInfo['LONG_GAGE']
dims = {}
dims['res'] = 0.125
dims['minlon'] = -81.0625
dims['maxlon'] = -67.4375
dims['minlat'] = 39.9375
dims['maxlat'] = 48.0625

def latlon2glatglon(lat, lon):
    glat = (lat - dims['minlat'])/dims['res'] + 0.5
    glon = (lon - dims['minlon'])/dims['res'] + 0.5
    return int(glat), int(glon)

def kriging(siteLat, siteLon, GEVpar):   

    """
    Universal kriging

    Args:
        :siteLat (array): latitude for the gauges 
        :siteLon (array): longitude for the gauges 
    """

    ### Setting the  prediction grid properties
    nlon = 110
    nlat = 66

    ### Read files
    mask = np.fromfile('./mask_NEUS.bin', 'float32').reshape(nlat, -1)
    mask = mask.reshape(nlon*nlat, -1)
    dem = np.fromfile('./gtopomean_NEUS.bin', 'float32').reshape(nlat, -1)
    ### Extract the colocated dem for gauges
    siteIndex = np.array([latlon2glatglon(siteLat[i], siteLon[i]) for i in range(len(siteLat))])
    siteDem = dem[siteIndex[:,0], siteIndex[:,1]]
    dem = dem.reshape(nlon*nlat, -1)

    ### Convert numpy to R format
    r.assign('cellsize', dims['res'])
    r.assign('minLat', dims['minlat'])
    r.assign('maxLat', dims['maxlat'])
    r.assign('minLon', dims['minlon'])
    r.assign('maxLon', dims['maxlon'])
    r.assign('siteLat', FloatVector(siteLat))
    r.assign('siteLon', FloatVector(siteLon))
    r.assign('siteDem', FloatVector(siteDem))
    r.assign('GEVpar', FloatVector(GEVpar))
    r.assign('mask', FloatVector(mask))
    r.assign('dem', FloatVector(dem))

    ### Import R packages
    importr('gstat')
    importr('sp')
    importr('automap')

    ### Create the grid for spatial predictions
    r("grd <- expand.grid(lon=seq(from=minLon, to=maxLon, by=cellsize), lat=seq(from=minLat, to=maxLat, by=cellsize))")
    r("data.site <- data.frame(lon=siteLon, lat=siteLat, GEVpar=GEVpar, dem=siteDem)")
    r("data.grid <- data.frame(lon=grd$lon, lat=grd$lat, mask=mask, dem=dem)")
    r("select <- which(data.grid$mask != -9.99e+08)")
    r("grd.mask <- data.frame(grd[select,], dem=data.grid$dem[select])")
    r("coordinates(data.site) <- ~lon+lat")
    r("coordinates(grd.mask) <- ~lon+lat")

    ##### Automaticly fit the variogram using 'automap' package and do the universal kriging
    r("data.kriged <- autoKrige(GEVpar~lon+lat+dem, data.site, grd.mask)")
    r("data.grid$kg.pred <- -9.99e+08")
    r("data.grid$kg.pred[select] <- data.kriged$krige_output$var1.pred")
    kgpred = np.array(r("data.grid$kg.pred")).reshape(nlat, nlon)

    '''
    importr('scatterplot3d')
    importr('lattice')
    r.pdf(file='./temp.pdf', width=4, height=4)
    r("plot(grd.mask)")
    r("plot(test)")
    #r("scatterplot3d(data.kriged$lon, data.kriged$lat, data.kriged$GEVpar.pred)")
    #r("image(data.kriged, loc = grd.mask)")
    #r("levelplot(data.kriged$GEVpar.pred~lon+lat, data=data.kriged)")
    r("dev.off()")
    '''

    return np.array(kgpred)

aa = kriging(gageLat.values, gageLon.values, gageDrainage)   

plt.figure()
M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)    # NEUS
M.scatter(gageLon, gageLat, 80, gageDrainage, cmap='Spectral_r', vmax=3600)
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
M.etopo()

dem = np.fromfile('./gtopomean_NEUS.bin','float32').reshape(66, -1)
mask = np.fromfile('./mask_NEUS.bin','float32').reshape(66, -1)
mask_NEUS = np.ma.masked_equal(mask, -9.99e+08).mask
dem_mask = np.ma.array(dem, mask=np.resize(mask_NEUS, dem.shape))
plt.figure()
plt.imshow(dem_mask[::-1])
plt.show()


