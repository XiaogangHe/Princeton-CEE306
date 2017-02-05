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

def cal_runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def get_annual_max(gageID, gageStime):

    """
    Get the annumal maximum flow for each station in the water year (Oct-Sep)
    """

    gageID_new = '0%s' % (gageID)
    data = pd.read_csv('%s/usgs_daily_%s.dat' % (data_dir, gageID_new), names=['flow'], sep=" ", header=None, skip_blank_lines=False)
    data = data.replace('Ice',np.nan)
    data = data.replace('Eqp',np.nan)
    data = data.replace(' ',np.nan)
    stime = '%s-10-01' % (gageStime)
    etime = '2015-12-31'
    date = pd.date_range(stime, etime, freq='D')
    data = Series(data.flow.astype('float').values, index=date)
    data_mon = data.resample("M", how='max')
    flow_max = np.nanmax(data_mon.values[:-3].reshape(-1,12), axis=1) ### May have NaN value

    return flow_max

def get_Qn(gageID, gageStime, n):

    """
    Get the n-day low flow for each station in the water year (Oct-Sep)
    """

    gageID_new = '0%s' % (gageID)
    data = pd.read_csv('%s/usgs_daily_%s.dat' % (data_dir, gageID_new), names=['flow'], sep=" ", header=None, skip_blank_lines=False)
    data = data.replace('Ice',np.nan)
    data = data.replace('Eqp',np.nan)
    data = data.replace(' ',np.nan)
    data = data.astype('float')
    data = cal_runningMean(data.flow.values, n)
    stime = '%s-10-01' % (gageStime)
    etime = '2015-12-%s' % (32-n)
    date = pd.date_range(stime, etime, freq='D')
    data = Series(data, index=date)
    data_mon = data.resample("M", how='min')
    flow_min = np.nanmin(data_mon.values[:-3].reshape(-1,12), axis=1) ### May have NaN value

    return flow_min

def TFPW_detect(ID):   

    """
    Use the trend-free pre-whitening (TFPW) method to detect the trend
    """

    importr('zyp')
    data_valid = gage_Q30[ID]
    data_valid = data_valid[~np.isnan(data_valid)]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)
    result = r("result <- zyp.trend.vector(ts, method='yuepilon', conf.intervals=TRUE)")
    trend = result[1]
    pvalue = result[5]

    return trend, pvalue

def fit_GEV(ID, stationary=True):   

    """
    Fit the GEV distribution using R package "ismev"
    """

    importr('ismev')
    importr('extRemes')
    data_valid = gage_Q30[ID]
    data_valid = data_valid[~np.isnan(data_valid)]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)

    ### For low flow, take the negative value of the input
    if stationary == True:
        r("fit_obj <- gev.fit(-ts)") 
        #r("fit_obj <- gev.fit(ts)") 
        pars = r("fit_obj$mle")    # location, scale, shape

        '''
        r.pdf(file='%s/gev_diag_Q7_%s_stationary_normQ.pdf' % (fig_dir, ID))
        r("gev.diag(fit_obj)")
        #r("fit_obj$cov[1,2] = -fit_obj$cov[1,2]")
        #r("fit_obj$cov[2,1] = -fit_obj$cov[2,1]")
        #r("fit_obj$cov[1,3] = -fit_obj$cov[1,3]")
        #r("fit_obj$cov[3,1] = -fit_obj$cov[3,1]")
        #r("gev.rl(c(-fit_obj$mle[1], fit_obj$mle[2:3]), fit_obj$cov, -fit_obj$data)")
        #r("gev.rl(fit_obj$mle, fit_obj$cov, fit_obj$data)")
        r("dev.off()")
        '''

        return -pars[0], pars[1], pars[2]
        
    else:
        #r("fit_obj <- gev.fit(-ts, ydat=matrix(1:length(ts), ncol=1), mul=1, sigl=1, siglink=exp)")
        r("fit_obj <- gev.fit(-ts, ydat=matrix(1:length(ts), ncol=1), mul=1, sigl=1)")
        pars = r("fit_obj$mle")    # location, scale, shape
        
        '''
        r.pdf(file='%s/gev_diag_Q7_%s_nonstationary_normQ.pdf' % (fig_dir, ID), width=8, height=4)
        r("gev.diag(fit_obj)")
        r("dev.off()")
        '''

        return -pars[0], -pars[1], pars[2], pars[3], pars[4]

def fit_GEV_climatic(ID, cliIndices):   

    """
    Fit the GEV distribution using R package "ismev"
    with climatic indexes
    """

    importr('ismev')
    importr('extRemes')
    data = gage_Q30[ID]
    ind = ~np.isnan(data)
    data_valid = data[ind]
    cliIndices_valid = cliIndices[ind]
    ts = FloatVector(data_valid)
    cliIndices_valid = FloatVector(cliIndices_valid)
    r.assign('ts', ts)
    r.assign('cliIndices_valid', cliIndices_valid)
    r("fit_obj <- gev.fit(-ts, ydat=matrix(cliIndices_valid, ncol=1), mul=1, sigl=1)")
    pars = r("fit_obj$mle")    # location, scale, shape

    return -pars[0], -pars[1], pars[2], pars[3], pars[4]

def fitGEV_LRT(ID, covariate):   

    """
    Fit the GEV distribution using R package "extRemes"
    with climatic indexes and conduct likelihood ratio test (LRT)

    Args:
        :covariate (array): covariate for non-stationary GEV parameters 
    """

    importr('ismev')
    importr('extRemes')
    data = gage_Q30[ID]
    ind = ~np.isnan(data)
    data_valid = data[ind]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)
    covariate_valid = covariate[ind]
    covariate_valid = FloatVector(covariate_valid)
    r.assign('covariate_valid', covariate_valid)

    r("fit0 <- fevd(-ts, type=c('GEV'))")
    r("fit11 <- fevd(-ts, location.fun=~covariate_valid, type=c('GEV'))")
    r("fit12 <- fevd(-ts, scale.fun=~covariate_valid, type=c('GEV'))")
    r("fit2 <- fevd(-ts, location.fun=~covariate_valid, scale.fun=~covariate_valid, type=c('GEV'))")

    ### Likelihood ratio test
    pvalue_mu1 = r("lr.test(fit0, fit11)$p.value")
    pvalue_sigma1 = r("lr.test(fit0, fit12)$p.value")

    ### Time series of the fitted parameters
    location_ts = r("location_ts <- -findpars(fit2)$location")    # For low flow () 
    scale_ts = r("scale_ts <- findpars(fit2)$scale")
    shape_ts = r("findpars(fit2)$shape")
    r("fit_mu <- lm(location_ts~covariate_valid)")
    r("fit_sigma <- lm(scale_ts~covariate_valid)")
    mu_0 = r("fit_mu$coefficients[[1]]")
    mu_1 = r("fit_mu$coefficients[[2]]")
    sigma_0 = r("fit_sigma$coefficients[[1]]")
    sigma_1 = r("fit_sigma$coefficients[[2]]")

    return location_ts, scale_ts, shape_ts, pvalue_mu1, pvalue_sigma1, mu_0, mu_1, sigma_0, sigma_1, covariate_valid 

def UKED(siteLat, siteLon, GEVpar):

    """
    Universal Kriging with External Drift

    Args:
        :siteLat (array): latitude for the gauges 
        :siteLon (array): longitude for the gauges 
        :GEVpar (array): GEV parameters 
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
    r("data.kriged <- autoKrige(GEVpar~lon+lat, data.site, grd.mask)")
    r("data.grid$kg.pred <- -9.99e+08")
    r("data.grid$kg.pred[select] <- data.kriged$krige_output$var1.pred")
    kgpred = np.array(r("data.grid$kg.pred")).reshape(nlat, nlon)

    return np.array(kgpred)

def NSUKED(gageID_select):
    """
    Non-stationary Universal Kriging with External Drift

    Args:
        :gageID_select (core.series.Series): selected gage ID with long-term records 
    """

    ### Examine whether the time series is stationary or not
    nSelect = gageID_select.shape[0]
    pvalue_TFPW = np.array([TFPW_detect('0%s'%(gageID_select.values[i]))[1] for i in range(nSelect)])       ## p-value from TFPW test for each station
    indSelect = gageID_select.index

    for i in range(nSelect):
        if pvalue_TFPW[i] <= 0.05:
            gageInfo['mu0'][indSelect[i]] = fit_GEV('0%s' % (gageID_select.values[i]), stationary=True)

        else:
            fit_GEV('0%s' % (gageID_select.values[i]), stationary=False)



    ### Examine whether the time series is stationary or not
    
    return

'''
### Test the .dat file
for i in range(gageID.shape[0]):
    print gageID[i]
    get_annual_max(gageID[i], gageStime[i])
'''

### Get climatic indices
SOI = np.loadtxt('%s/SOI.latest' % (clidir), skiprows=1) 
PDO = np.loadtxt('%s/PDO.latest' % (clidir), skiprows=1) 
AMO = np.loadtxt('%s/AMO.latest' % (clidir), skiprows=1) 
SOI_Jan = SOI[27:-1, 1]    # Just an example

### Get gauge data
gage_Q7 = {'0%s' % (gageID[i]): get_Qn(gageID[i], gageStime[i], 7) for i in range(gageID.shape[0])}
gage_Q30 = {'0%s' % (gageID[i]): get_Qn(gageID[i], gageStime[i], 30) for i in range(gageID.shape[0])}
gageValidLen = np.array([np.sum(~np.isnan(gage_Q30['0%s' % (gageID[i])])) for i in range(gageID.shape[0])])
gageInfo['ValidLen'] = gageValidLen
index = gageInfo['ValidLen'] > minLen
gageID_long = gageID[index]
lat = gageInfo['LAT_GAGE'][index]
lon = gageInfo['LONG_GAGE'][index]
gageInfo['mu0'] = np.nan
gageInfo['mu1'] = np.nan
gageInfo['sigma0'] = np.nan
gageInfo['sigma1'] = np.nan
gageInfo['xi'] = np.nan

### The first station
fitGEV_LRT('01013500', SOI_Jan)
fitGEV_LRT('01013500', np.arange(gageStime[0], 2015))

### LRT for all stations
#pvalue_mu = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), SOI[gageStime[gageID_long.index].values[i]-1876:-1, 10:13].mean(-1))[3] for i in range(gageID_long.shape[0])])
#pvalue_sigma = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), SOI[gageStime[gageID_long.index].values[i]-1876:-1, 10:13].mean(-1))[4] for i in range(gageID_long.shape[0])])
#pvalue_mu = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), PDO[gageStime[gageID_long.index].values[i]-1900:-1, 10:13].mean(-1))[3] for i in range(gageID_long.shape[0])])
#pvalue_sigma = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), PDO[gageStime[gageID_long.index].values[i]-1900:-1, 10:13].mean(-1))[4] for i in range(gageID_long.shape[0])])
pvalue_mu = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), AMO[gageStime[gageID_long.index].values[i]-1857:-1, 7:10].mean(-1))[3] for i in range(gageID_long.shape[0])])
pvalue_sigma = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), AMO[gageStime[gageID_long.index].values[i]-1857:-1, 7:10].mean(-1))[4] for i in range(gageID_long.shape[0])])

#pvalue_mu = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), np.arange(gageStime[gageID_long.index].values[i],2015))[3] for i in range(gageID_long.shape[0])])
#pvalue_sigma = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), np.arange(gageStime[gageID_long.index].values[i],2015))[-1] for i in range(gageID_long.shape[0])])

### UKED interpolation
UKED_pred = UKED(gageLat.values, gageLon.values, gageDrainage)

### Plot the p-value for mu1 and sigma1
ind_pval = pvalue_mu < 0.05
#ind_pval = pvalue_sigma < 0.05
plt.figure()
M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)    # NEUS
#M = Basemap(resolution='l', llcrnrlat=33, urcrnrlat=42, llcrnrlon=-95, urcrnrlon=-75)   # EastHghlnds
M.scatter(lon[ind_pval.squeeze()], lat[ind_pval.squeeze()], color='k', s=150, marker='.')
M.scatter(lon[~ind_pval.squeeze()], lat[~ind_pval.squeeze()], color='r', s=150, marker='.')
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
M.etopo()
#plt.savefig('%s/LST_pvalue_mu1_AMO_JAS_Q30.pdf' % (fig_dir))
#plt.savefig('%s/LST_pvalue_sigma1_AMO_JAS_Q30.pdf' % (fig_dir))
plt.show()

"""
### Get the p-value from the TFPW test
ps = np.array([TFPW_detect('0%s'%(gageID_long.values[i]))[1] for i in range(gageID_long.shape[0])])    ## p-value from TFPW test for each station
trend = np.array([TFPW_detect('0%s'%(gageID_long.values[i]))[0] for i in range(gageID_long.shape[0])])    ## trend from TFPW test for each station
ind = ps<=0.05
ind2 = ps>0.05

### Calculate the GEV parameters for the STATIONARY case
location = np.array([fit_GEV('0%s' % (gageID_long.values[i]), stationary=True)[0] for i in range(gageID_long.shape[0])])
scale = np.array([fit_GEV('0%s' % (gageID_long.values[i]), stationary=True)[1] for i in range(gageID_long.shape[0])])
shape = np.array([fit_GEV('0%s' % (gageID_long.values[i]), stationary=True)[2] for i in range(gageID_long.shape[0])])

### Calculate the GEV parameters for the NON-STATIONARY case
par_total = np.array([fit_GEV('0%s' % (gageID_long.values[i]), stationary=False) for i in range(gageID_long.shape[0])])
mu0 = par_total[:, 0]
mu1 = par_total[:, 1]
sigma0 = par_total[:, 2]
sigma1 = par_total[:, 3]
xi = par_total[:, 4]

### Calculate the trend
p1 = 0.01
yp1 = mu1-sigma1/xi*(1-np.power(-np.log(1-p1),-xi))

'''
### Plot the relationship between GEV parameters and other covariates
plt.figure()
plt.scatter(gageDrainage[index], location)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Drainage area (km2)')
plt.ylabel('Location [cfs]')

plt.figure()
plt.scatter(gageDrainage[index], scale)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Drainage area (km2)')
plt.ylabel('Scale [-]')

plt.figure()
plt.scatter(gageDrainage[index], shape)
plt.xscale('log')
plt.xlabel('Drainage area (km2)')
plt.ylabel('Shape [-]')
plt.show()

### Diagnosis plot for each site
for i in range(gageID_long.shape[0]):
    if ps[i] <= 0.05:
        fit_GEV('0%s' % (gageID_long.values[i]), stationary=True)
        fit_GEV('0%s' % (gageID_long.values[i]), stationary=False)
    else:
        fit_GEV('0%s' % (gageID_long.values[i]), stationary=True)

### Plot the trend
plt.figure()
#M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)    # NEUS
M = Basemap(resolution='l', llcrnrlat=33, urcrnrlat=42, llcrnrlon=-95, urcrnrlon=-75)   # EastHghlnds
M.scatter(lon[ind], lat[ind], color='k', s=150, marker='.')
M.scatter(lon[ind2], lat[ind2], color='r', s=150, marker='.')
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
M.etopo()
#plt.savefig('%s/HCDN_sites_EastHghlnds_trend_Q30.pdf' % (fig_dir))
#plt.savefig('%s/HCDN_sites_NEUS_trend_Q30.pdf' % (fig_dir))
plt.show()
'''

plt.figure()
M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)    # NEUS
#M = Basemap(resolution='l', llcrnrlat=33, urcrnrlat=42, llcrnrlon=-95, urcrnrlon=-75)   # EastHghlnds
sc = M.scatter(lon[ind], lat[ind], 200, yp1[ind], marker='.', cmap='coolwarm')
M.scatter(lon[ind2], lat[ind2], color='k', s=50, marker='^')
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
#M.etopo()
M.shadedrelief(scale=0.5)
cb = M.colorbar(sc)
cb.solids.set_edgecolor("face")
#plt.title('100-year return level of $Q_{7}$')
#plt.savefig('%s/HCDN_sites_EastHghlnds_trend_Q30.pdf' % (fig_dir))
#plt.savefig('%s/HCDN_sites_NEUS_trend_Q30_Q100.pdf' % (fig_dir))
plt.show()

plt.figure()
M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)    # NEUS
#M = Basemap(resolution='l', llcrnrlat=33, urcrnrlat=42, llcrnrlon=-95, urcrnrlon=-75)   # EastHghlnds
sc = M.scatter(lon[ind], lat[ind], 200, trend[ind], marker='.', cmap='Spectral_r')
M.scatter(lon[ind2], lat[ind2], color='k', s=50, marker='^')
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
#M.etopo()
M.shadedrelief(scale=0.5)
cb = M.colorbar(sc)
cb.solids.set_edgecolor("face")
#plt.title('$Q_{7}$')
#plt.savefig('%s/HCDN_sites_EastHghlnds_trend_Q30.pdf' % (fig_dir))
#plt.savefig('%s/HCDN_sites_NEUS_trend_Q30_new.pdf' % (fig_dir))
plt.show()
"""
