#!/usr/bin/env python

import numpy as np
import matplotlib
from pylab import rcParams
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import Array as rArray
import rpy2.robjects.numpy2ri as rpyn
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import geopandas as gp

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
data_dir = '../HCDN/NEUS'
fig_dir = '/home/wind/hexg/Dropbox/Research_Princeton/GeneralExam/Results'
rep_dir = '../USGS_GageII/basinchar_and_report_sept_2011'
bas_dir = '../USGS_GageII/boundaries-shapefiles-by-aggeco'                            # Basin shapefile
minLen = 60
gageInfo = pd.read_excel('%s/HCDN-2009_Station_Info_NEUS.xlsx' % (data_dir))
gageID = gageInfo['STATION ID']
gageStime = gageInfo['stime (WY)']
gageDrainage = gageInfo['DRAIN_SQKM']
gageLat = gageInfo['LAT_GAGE']
gageLon = gageInfo['LONG_GAGE']

def cal_runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

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
    data_valid = gage_Q7[ID]
    data_valid = data_valid[~np.isnan(data_valid)]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)
    result = r("result <- zyp.trend.vector(ts, method='yuepilon', conf.intervals=TRUE)")
    trend = result[1]
    pvalue = result[5]

    return trend, pvalue

def fitGEV(ID, stationary=True):   

    """
    Fit the GEV distribution using R package "ismev"
    """

    importr('ismev')
    importr('extRemes')
    data_valid = gage_Q7[ID]
    data_valid = data_valid[~np.isnan(data_valid)]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)

    if stationary == True:
        r("fit.obj <- gev.fit(-ts)")    # For low flow, take the negative value of the input 
        #r("fit_obj <- gev.fit(ts)")    # For max flow 
        pars = r("fit.obj$mle")         # location, scale, shape
        return -pars[0], pars[1], pars[2]
        
    else:
        r("fit.obj <- gev.fit(-ts, ydat=matrix(1:length(ts), ncol=1), mul=1, sigl=1)")
        #r("fit.obj <- gev.fit(-ts, ydat=matrix(1:length(ts), ncol=1), mul=1, sigl=1, siglink=exp)")    # log(sigma(t))=sigma0+sigma1*t
        pars = r("fit.obj$mle")         # location, scale, shape
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
    data = gage_Q7[ID]
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
    r("fit2 <- fevd(-ts, location.fun=~covariate_valid, scale.fun=~covariate_valid, type=c('GEV'))")    # use.phi=TRUE for log(sigma)

    ### Likelihood ratio test
    pv_mu1 = r("lr.test(fit0, fit11)$p.value")
    pv_sigma1 = r("lr.test(fit0, fit12)$p.value")

    ### Time series of the fitted parameters
    location_ts = r("location_ts <- -findpars(fit2)$location")    # For low flow 
    scale_ts = r("scale_ts <- findpars(fit2)$scale")

    print "#####"
    print scale_ts

    shape_ts = r("findpars(fit2)$shape")
    r("fit_mu <- lm(location_ts~covariate_valid)")
    r("fit_sigma <- lm(scale_ts~covariate_valid)")
    mu0 = r("fit_mu$coefficients[[1]]")
    mu1 = r("fit_mu$coefficients[[2]]")
    sigma0 = r("fit_sigma$coefficients[[1]]")
    sigma1 = r("fit_sigma$coefficients[[2]]")

    return location_ts, scale_ts, shape_ts, pv_mu1, pv_sigma1, mu0, mu1, sigma0, sigma1, covariate_valid

def NSGEV(gageID_select):
    """
    Non-stationary GEV

    Args:
        :gageID_select (core.series.Series): selected gage ID with long-term records 
    """

    ### Examine whether the time series is stationary or not
    nSelect = gageID_select.shape[0]
    pv_trend = np.array([TFPW_detect('0%s'%(gageID_select.values[i]))[1] for i in range(nSelect)])       ## p-value from TFPW test 
    indSelect = gageID_select.index

    for i in range(nSelect):
        #cov_test = SOI[gageStime[gageID_select.index].values[i]-1876:-1, 4:7].mean(-1) 
        cov_test = np.arange(gageStime[gageID_select.index].values[i], 2015)
        if pv_trend[i] > 0.05:
            print "stationary station"
            mu0, sigma0, xi = fitGEV('0%s' % (gageID_select.values[i]), stationary=True)
            gageInfo['mu0'][indSelect[i]] = mu0
            gageInfo['mu1'][indSelect[i]] = 0
            gageInfo['sigma0'][indSelect[i]] = sigma0
            gageInfo['sigma1'][indSelect[i]] = 0
            gageInfo['xi'][indSelect[i]] = xi
        else:
            xi, pv_mu1, pv_sigma1, mu0, mu1, sigma0, sigma1 = fitGEV_LRT('0%s' % (gageID_select.values[i]), cov_test)[2:-1]
            gageInfo['mu0'][indSelect[i]] = np.array(mu0)
            gageInfo['mu1'][indSelect[i]] = np.array(mu1)
            gageInfo['sigma0'][indSelect[i]] = np.array(sigma0)
            gageInfo['sigma1'][indSelect[i]] = np.array(sigma1)
            gageInfo['xi'][indSelect[i]] = np.array(xi)[0]
            ### Examine whether mu1 and sigma1 is significantly different from 0
            if np.array(pv_mu1) > 0.05:
                gageInfo['mu1'][indSelect[i]] = 0
            if np.array(pv_sigma1) > 0.05:
                gageInfo['sigma1'][indSelect[i]] = 0

    return

def UK_basin(centroid_lat_obs, centroid_lon_obs, centroid_lat_pre, centroid_lon_pre, GEVpar):

    """
    Universal Kriging for each basin 

    Args:
        :centroid_lat_obs (array): lat of centroid location for known basins 
        :centroid_lon_obs (array): lon of centroid location for known basins 
        :centroid_lat_pre (array): lat of centroid location for unknown basins 
        :centroid_lon_pre (array): lon of centroid location for unknown basins 
        :GEVpar (array): GEV parameters 
    """

    ### Convert numpy to R format
    r.assign('centroid_lat_obs', FloatVector(centroid_lat_obs))
    r.assign('centroid_lon_obs', FloatVector(centroid_lon_obs))
    r.assign('centroid_lat_pre', FloatVector(centroid_lat_pre))
    r.assign('centroid_lon_pre', FloatVector(centroid_lon_pre))
    r.assign('GEVpar', FloatVector(GEVpar))

    ### Import R packages
    importr('gstat')
    importr('sp')
    importr('automap')

    ### Create the grid for spatial predictions
    r("data.obs <- data.frame(lon=centroid_lon_obs, lat=centroid_lat_obs, GEVpar=GEVpar)")
    r("data.pre <- data.frame(lon=centroid_lon_pre, lat=centroid_lat_pre)")
    r("coordinates(data.obs) <- ~lon+lat")
    r("coordinates(data.pre) <- ~lon+lat")

    ### Automaticly fit the variogram using 'automap' package and do the universal kriging
    r("data.kriged <- autoKrige(GEVpar~lon+lat, data.obs, data.pre)")
    kg_pred = np.array(r("data.kriged$krige_output$var1.pred"))
    kg_stdev = np.array(r("data.kriged$krige_output$var1.stdev"))

    ### Cross validation
    r("uk.cv <- autoKrige.cv(GEVpar~lon+lat, data.obs, nfold=nrow(data.obs))")
    kg_cv_error = np.array(r("uk.cv$krige.cv_output$residual"))
    print r("uk.cv$krige.cv_output[,'residual']")
    np.savetxt('./cv_residual_OK.txt', kg_cv_error, fmt='%.5f')
    print r("cv_out <- summary(uk.cv)")

    ### Plot the CV error
    r.pdf(file='../Figures/OK_HCDN_CV_error.pdf')
    r("compare.cv(uk.cv, bubbleplots = TRUE, col.names = c('UK'))")
    r("dev.off()")

    return np.array(kg_pred), np.array(kg_stdev), np.array(kg_cv_error)

def KED_basin(centroid_lat_obs, centroid_lon_obs, cov_obs, centroid_lat_pre, centroid_lon_pre, cov_pre, GEVpar):

    """
    Kriging with external drift for each basin 

    Args:
        :centroid_lat_obs (array): lat of centroid location for known basins 
        :centroid_lon_obs (array): lon of centroid location for known basins 
        :centroid_lat_pre (array): lat of centroid location for unknown basins 
        :centroid_lon_pre (array): lon of centroid location for unknown basins 
        :cov_obs (array): additional covariates of known basins 
        :cov_pre (array): additional covariates of unknown basins 
        :GEVpar (array): GEV parameters 
    """

    ### Convert numpy to R format
    r.assign('centroid_lat_obs', FloatVector(centroid_lat_obs))
    r.assign('centroid_lon_obs', FloatVector(centroid_lon_obs))
    r.assign('centroid_lat_pre', FloatVector(centroid_lat_pre))
    r.assign('centroid_lon_pre', FloatVector(centroid_lon_pre))
    r.assign('cov1_obs', FloatVector(cov_obs[:,0]))
    r.assign('cov2_obs', FloatVector(cov_obs[:,1]))
    r.assign('cov3_obs', FloatVector(cov_obs[:,2]))
    r.assign('cov4_obs', FloatVector(cov_obs[:,3]))
    r.assign('cov5_obs', FloatVector(cov_obs[:,4]))
    r.assign('cov1_pre', FloatVector(cov_pre[:,0]))
    r.assign('cov2_pre', FloatVector(cov_pre[:,1]))
    r.assign('cov3_pre', FloatVector(cov_pre[:,2]))
    r.assign('cov4_pre', FloatVector(cov_pre[:,3]))
    r.assign('cov5_pre', FloatVector(cov_pre[:,4]))
    r.assign('GEVpar', FloatVector(GEVpar))

    ### Import R packages
    importr('gstat')
    importr('sp')
    importr('automap')

    ### Create the grid for spatial predictions
    r("data.obs <- data.frame(lon=centroid_lon_obs, lat=centroid_lat_obs, cov1=cov1_obs, cov2=cov2_obs, cov3=cov3_obs, cov4=cov4_obs, cov5=cov5_obs, GEVpar=GEVpar)")
    r("data.pre <- data.frame(lon=centroid_lon_pre, lat=centroid_lat_pre, cov1=cov1_pre, cov2=cov2_pre, cov3=cov3_pre, cov4=cov4_pre, cov5=cov5_pre)")
    r("coordinates(data.obs) <- ~lon+lat")
    r("coordinates(data.pre) <- ~lon+lat")

    ### Automaticly fit the variogram using 'automap' package and do the universal kriging
    r("data.kriged <- autoKrige(GEVpar~lon+lat+cov1+cov2+cov3, data.obs, data.pre)")
    kg_pred = np.array(r("data.kriged$krige_output$var1.pred"))
    kg_stdev = np.array(r("data.kriged$krige_output$var1.stdev"))

    ### Cross validation
    r("ked.cv <- autoKrige.cv(GEVpar~lon+lat+cov1+cov2+cov3, data.obs, nfold=nrow(data.obs))")
    kg_cv_error = np.array(r("ked.cv$krige.cv_output$residual"))
    np.savetxt('./cv_residual_KED.txt', kg_cv_error, fmt='%.5f')
    print r("cv_out <- summary(ked.cv)")

    ### Plot the CV error
    r.pdf(file='../Figures/KED_HCDN_CV_error.pdf')
    r("compare.cv(ked.cv, bubbleplots = TRUE, col.names = c('KED'))")
    r("dev.off()")

    return np.array(kg_pred), np.array(kg_stdev), np.array(kg_cv_error)

def cal_RP_basin(GEVpar_loc, GEVpar_sca, GEVpar_shp):

    """
    Calculate the return period for each basin based on the GEV parameters from krigging

    Args:
        :GEVpar_loc: location parameter 
        :GEVpar_sca: scale parameter 
        :GEVpar_shp: shape parameter 
    """

    ### Convert numpy to R format
    r.assign('GEVpar_loc', GEVpar_loc)
    r.assign('GEVpar_sca', GEVpar_sca)
    r.assign('GEVpar_shp', GEVpar_shp)

    ### Import R packages
    importr('extRemes')

    ### Calculate the return period given the GEV parameters
    #r("rp.obj <- rlevd(c(2, 20, 50, 100), loc=GEVpar_loc, scale=GEVpar_sca, shape=GEVpar_shp, type=c('GEV'))")    ### method 1
    r("z <- revd(1000, loc=-GEVpar_loc, scale=GEVpar_sca, shape=GEVpar_shp, type = c('GEV'))")    # Take negtive value of location for low flow
    r("fit.obj <- fevd(z, method='Lmoments')")
    r("rl.obj <- return.level(fit.obj, return.period=c(2,20,50,100), alpha=0.1, verbose=TRUE, do.ci=TRUE)")    # 95% CI

    ### Take the negative value for different return level 
    rl2 = -np.array(r("rl.obj['2-year',]"))
    rl20 = -np.array(r("rl.obj['20-year',]"))
    rl50 = -np.array(r("rl.obj['50-year',]"))
    rl100 = -np.array(r("rl.obj['100-year',]"))

    return rl2, rl20, rl50, rl100

def cal_ReturnLevelTrend(mu1, sigma1, xi, p=0.01):
    yp_trend = mu1-sigma1/xi*(1-np.power(-np.log(1-p),-xi))
    return yp_trend

def plot_kriging_pred(GEV_kgpred, GEV_obs, var_name, kriging_method):
    """
    Plot the Kriging prediction 

    Args:
        :GEV_kgpred (array): kriging prediction  
    """

    basin_GEV_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_GEV_2d[basin_GEV_2d==iBasin_FID] = GEV_kgpred[i]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    mcolors = plt.cm.Spectral_r(np.linspace(0.1, 0.6, 100))
    cmap_pred = colors.ListedColormap(mcolors)

    m = Basemap(projection='mill', llcrnrlon=-82.953, llcrnrlat=37.853, urcrnrlon=-66.395, urcrnrlat=49.166, suppress_ticks=True)
    cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=GEV_obs.min(), vmax=GEV_obs.max(), cmap=cmap_pred, alpha=0.8) 
    cb = plt.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")
    plt.title("$\%s$, %s predictions" % (var_name, kriging_method), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_NS_kriging_pred_LRT(GEV_kgpred, GEV_obs, pv_trend, pv_GEVpar, point_locs, var_name, kriging_method):
    """
    Plot the Kriging prediction for non-stationary GEV & LRT tests 

    Args:
        :GEV_kgpred (array): kriging prediction for each basin 
        :GEV_obs (array): GEV parameter for gauges
        :pv_trend (array): p-value for the trend detection
        :pv_GEVpar (array): p-value for the likelihood ratio test (mu1 and sigma1)
    """

    basin_GEV_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_GEV_2d[basin_GEV_2d==iBasin_FID] = GEV_kgpred[i]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    mcolors = plt.cm.Spectral_r(np.linspace(0.1, 0.6, 100))
    cmap_pred = colors.ListedColormap(mcolors)

    m = Basemap(projection='mill', llcrnrlon=-82.953, llcrnrlat=37.853, urcrnrlon=-66.395, urcrnrlat=49.166, suppress_ticks=True)
    cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=GEV_obs.min(), vmax=GEV_obs.max(), cmap=cmap_pred, alpha=0.8) 
    cb = plt.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")

    ind1 = pv_trend > 0.05                                   # Stationary             
    ind2 = (pv_GEVpar.squeeze()<=0.05) & (pv_trend<=0.05)    # Non-stationary and significantly different from 0
    ind3 = (pv_GEVpar.squeeze()>0.05) & (pv_trend<=0.05)     # Non-stationary but not significantly from 0 

    x, y = m(point_locs[:,0], point_locs[:,1])
    m.scatter(x[ind1], y[ind1], color='k', s=80, marker='^', label='Stationary')
    m.scatter(x[ind2], y[ind2], color='#ef4026', s=100, label='Significant')
    m.scatter(x[ind3], y[ind3], color='#06c2ac', s=100, label='Not-significant')
    plt.legend(loc=2)

    plt.title("$\%s$, %s predictions" % (var_name, kriging_method), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_kriging_stdev(GEV_kgstdev, var_name, kriging_method):

    basin_GEV_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_GEV_2d[basin_GEV_2d==iBasin_FID] = GEV_kgstdev[i]

    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    #cs = ax.imshow(np.ma.masked_equal(basin_GEV_2d, -9999), vmin=0.01, vmax=0.1, cmap=cm.Spectral_r)    # location, Need to change the color
    cs = ax.imshow(np.ma.masked_equal(basin_GEV_2d, -9999), vmin=0, vmax=0.05, cmap=cm.Spectral_r)     # scale
    #cs = ax.imshow(np.ma.masked_equal(basin_GEV_2d, -9999), vmin=0, vmax=0.15, cmap=plt.cm.Spectral_r) # shape
    cb = plt.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")
    plt.title("$\%s$, %s errors" % (var_name, kriging_method), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_kriging_stdev_CVerror(GEV_kgstdev, GEV_CVerror, point_locs, var_name, kriging_method):

    basin_GEV_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_GEV_2d[basin_GEV_2d==iBasin_FID] = GEV_kgstdev[i]

    #GEV_CVerror_abs_max = 0.25      # location
    #GEV_CVerror_abs_max = 3.4       # mu0
    #GEV_CVerror_abs_max = 0.1       # scale
    GEV_CVerror_abs_max = 2.5       # sigma0
    #GEV_CVerror_abs_max = 0.35      # shape
    size_max = 300
    size_pts = size_max*np.absolute(GEV_CVerror)/GEV_CVerror_abs_max

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    mcolors = plt.cm.Spectral_r(np.linspace(0.1, 0.9, 100))
    cmap_stdev = colors.ListedColormap(mcolors)

    m = Basemap(projection='mill', llcrnrlon=-82.953, llcrnrlat=37.853, urcrnrlon=-66.395, urcrnrlat=49.166, suppress_ticks=True)
    #cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=0.01, vmax=0.1, cmap=cmap_stdev, alpha=0.8)     # location
    #cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=1, vmax=2, cmap=cmap_stdev, alpha=0.8)          # mu0
    #cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=0, vmax=0.25, cmap=cmap_stdev, alpha=0.8)        # scale 
    cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=0.8, vmax=1.2, cmap=cmap_stdev, alpha=0.8)       # sigma0 
    #cs = m.imshow(np.ma.masked_equal(basin_GEV_2d[::-1], -9999.), vmin=0, vmax=0.15, cmap=cmap_stdev, alpha=0.8)       # shape
    cb = plt.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")

    x, y = m(point_locs[:,0], point_locs[:,1])
    m.scatter(x[GEV_CVerror>0], y[GEV_CVerror>0], size_pts[GEV_CVerror>0], facecolor='#10a674', alpha=0.8, marker='o', lw=.25, edgecolor='w')
    m.scatter(x[GEV_CVerror<0], y[GEV_CVerror<0], size_pts[GEV_CVerror<0], facecolor='#a50055', alpha=0.8, marker='o', lw=.25, edgecolor='w')
    l1 = plt.scatter([],[], s=100, edgecolors='none', c='#10a674')
    l2 = plt.scatter([],[], s=200, edgecolors='none', c='#10a674')
    l3 = plt.scatter([],[], s=300, edgecolors='none', c='#10a674')
    l4 = plt.scatter([],[], s=100, edgecolors='none', c='#a50055')
    l5 = plt.scatter([],[], s=200, edgecolors='none', c='#a50055')
    l6 = plt.scatter([],[], s=300, edgecolors='none', c='#a50055')

    labels_size = [GEV_CVerror_abs_max/i for i in range(1,4)]
    labels_pos = ["{:4.2f}".format(size) for size in labels_size][::-1]
    labels_neg = ["{:4.2f}".format(-size) for size in labels_size]

    leg1 = plt.legend([l4, l5, l6], labels_pos, handlelength=2, loc = 2, bbox_to_anchor=(0,0.8), borderpad = 1, handletextpad=0.5, scatterpoints = 1, frameon=False)
    leg2 = plt.legend([l3, l2, l1], labels_neg, handlelength=2, loc = 2, bbox_to_anchor=(0,1), borderpad = 1, handletextpad=0.5, scatterpoints = 1, frameon=False)
    plt.gca().add_artist(leg1)

    plt.title("$\%s$, %s errors" % (var_name, kriging_method), fontsize=25)
    plt.savefig('../Figures/temp.pdf', alpha=True)
    plt.show()

def plot_rl(rl, year, kriging_method):
    """
    Plot the return level estimated from GEV distribution 

    Args:
        :rl (array): return level  
        :year (str): for specific year  
    """

    basin_rl_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_rl_2d[basin_rl_2d==iBasin_FID] = rl[i]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    mcolors = plt.cm.YlOrRd(np.linspace(0.2, 1, 50))
    cmap_rl = colors.ListedColormap(mcolors)

    m = Basemap(projection='mill', llcrnrlon=-82.953, llcrnrlat=37.853, urcrnrlon=-66.395, urcrnrlat=49.166, suppress_ticks=True)
    cs = m.imshow(np.ma.masked_equal(basin_rl_2d[::-1], -9999.), vmin=0.0, vmax=0.4, cmap=cmap_rl)
    cb = fig.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(axis='y', direction='in')
    plt.title("%s-year return level, %s mean" % (year, kriging_method), fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_rltrend(rltrend, year, kriging_method):
    """
    Plot the trend for return level for non-stationary case 

    Args:
        :rl (array): return level  
        :year (str): for specific year  
    """

    basin_rltrend_2d = basin_FID_2d.copy()
    for i, iBasin_FID in enumerate(basin_FID_1d):
        basin_rltrend_2d[basin_rltrend_2d==iBasin_FID] = rltrend[i]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    mcolors = plt.cm.RdYlBu_r(np.linspace(0, 1, 100))
    cmap_rltrend = colors.ListedColormap(mcolors)

    m = Basemap(projection='mill', llcrnrlon=-82.953, llcrnrlat=37.853, urcrnrlon=-66.395, urcrnrlat=49.166, suppress_ticks=True)
    #cs = m.imshow(np.ma.masked_equal(basin_rltrend_2d[::-1], -9999.), vmin=-0.0008, vmax=0.002, cmap=cmap_rltrend)
    cs = m.imshow(np.ma.masked_equal(basin_rltrend_2d[::-1], -9999.), vmin=-0.0015, vmax=0.0015, cmap=cmap_rltrend)
    cb = fig.colorbar(cs, orientation='vertical')
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(axis='y', direction='in')
    plt.title("Trend of %s-year return level, %s" % (year, kriging_method), fontsize=25)
    plt.xticks([])
    plt.yticks([])

    plt.show()

def plot_hist_GEVpar(GEVpar_OK, GEVpar_KED):
    plt.figure()
    plt.hist(GEVpar_OK, 100, normed=True, label='OK', alpha=0.8, color='#0d3362')
    plt.hist(GEVpar_KED, 100, normed=True, label='KED', alpha=0.8, color='#c64737')
    plt.legend()
    plt.show()


### Get gauge data
gage_Q7 = {'0%s' % (gageID[i]): get_Qn(gageID[i], gageStime[i], 7)*2.83*1e-5*86400/gageDrainage[i] for i in range(gageID.shape[0])}    # mm/day
gage_Q30 = {'0%s' % (gageID[i]): get_Qn(gageID[i], gageStime[i], 30)*2.83*1e-5*86400/gageDrainage[i] for i in range(gageID.shape[0])}  # mm/day
gageValidLen = np.array([np.sum(~np.isnan(gage_Q7['0%s' % (gageID[i])])) for i in range(gageID.shape[0])])
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

'''
gageInfo['DEM_mean'] = np.nan
gageInfo['DEM_std'] = np.nan
gageInfo['SLP'] = np.nan                # slope
gageInfo['RR'] = np.nan                 # relief ratio
gageInfo['TOPWET'] = np.nan             # topographic wetness index
gageInfo['BFI'] = np.nan                # baseflow index
gageInfo['streams'] = np.nan            # stream density
gageInfo['sinuousity'] = np.nan         # sinuousity of mainstem streamline
gageInfo['contact'] = np.nan            # subsurface flow contact time index
gageInfo['SnowPct'] = np.nan            # snow percent of total precipitation
gageInfo['PrecSeaInd'] = np.nan         # precipitation seasonality index
gageInfo['AWCAVE'] = np.nan             # average available water capacity
gageInfo['PERMAVE'] = np.nan            # average permeability
gageInfo['BDAVE'] = np.nan              # average bulk density
gageInfo['OMAVE'] = np.nan              # average organic matter
gageInfo['WTDEPAVE'] = np.nan           # average depth to seasonally high water table

### Read the Gage ID from the basin shape file
basin_SHP = gp.GeoDataFrame.from_file('%s/bas_nonref_NorthEast_wgs84.shp' % (bas_dir))
basin_FID_2d = np.loadtxt('../USGS_GageII/GageII_NE_basin_raster_250m_wgs84.txt', skiprows=6)
basin_FID_1d = np.array(map(np.unique, basin_FID_2d.reshape(1,-1))).squeeze()[1:]
basin_GID_1d_all = basin_SHP['GAGE_ID']
basin_GID_1d_rst = basin_GID_1d_all[basin_FID_1d]    # Gage ID corresponding to the sampled raster file 

### Get GAGES II dataset
topo = pd.read_excel('%s/gagesII_sept30_2011_conterm.xlsx' % (rep_dir), sheetname='Topo', converters={'STAID': str})
topo_MEAN = np.array([topo[topo['STAID']=='0%s'%(gageID_long.values[i])]['ELEV_MEAN_M_BASIN'].values[0] for i in range(len(gageID_long))])
topo_MEAN_basin = np.array([topo[topo['STAID']==ibasin_GID]['ELEV_MEAN_M_BASIN'].values[0] for ibasin_GID in basin_GID_1d_rst])
topo_std = np.array([topo[topo['STAID']=='0%s'%(gageID_long.values[i])]['ELEV_STD_M_BASIN'].values[0] for i in range(len(gageID_long))])
topo_std_basin = np.array([topo[topo['STAID']==ibasin_GID]['ELEV_STD_M_BASIN'].values[0] for ibasin_GID in basin_GID_1d_rst])
topo_SLP = np.array([topo[topo['STAID']=='0%s'%(gageID_long.values[i])]['SLOPE_PCT'].values[0] for i in range(len(gageID_long))])
topo_RR = np.array([topo[topo['STAID']=='0%s'%(gageID_long.values[i])]['RRMEAN'].values[0] for i in range(len(gageID_long))])

hydro = pd.read_excel('%s/gagesII_sept30_2011_conterm.xlsx' % (rep_dir), sheetname='Hydro', converters={'STAID': str})
hydro_TOPWET = np.array([hydro[hydro['STAID']=='0%s'%(gageID_long.values[i])]['TOPWET'].values[0] for i in range(len(gageID_long))])
hydro_TOPWET_basin = np.array([hydro[hydro['STAID']==ibasin_GID]['TOPWET'].values[0] for ibasin_GID in basin_GID_1d_rst])
hydro_BFI = np.array([hydro[hydro['STAID']=='0%s'%(gageID_long.values[i])]['BFI_AVE'].values[0] for i in range(len(gageID_long))])
hydro_BFI_basin = np.array([hydro[hydro['STAID']==ibasin_GID]['BFI_AVE'].values[0] for ibasin_GID in basin_GID_1d_rst])
hydro_streams = np.array([hydro[hydro['STAID']=='0%s'%(gageID_long.values[i])]['STREAMS_KM_SQ_KM'].values[0] for i in range(len(gageID_long))])
hydro_sinuousity = np.array([hydro[hydro['STAID']=='0%s'%(gageID_long.values[i])]['MAINSTEM_SINUOUSITY'].values[0] for i in range(len(gageID_long))])
hydro_contact = np.array([hydro[hydro['STAID']=='0%s'%(gageID_long.values[i])]['CONTACT'].values[0] for i in range(len(gageID_long))])

clima = pd.read_excel('%s/gagesII_sept30_2011_conterm.xlsx' % (rep_dir), sheetname='Climate', converters={'STAID': str})
clima_SnowPct = np.array([clima[clima['STAID']=='0%s'%(gageID_long.values[i])]['SNOW_PCT_PRECIP'].values[0] for i in range(len(gageID_long))])
clima_SnowPct_basin = np.array([clima[clima['STAID']==ibasin_GID]['SNOW_PCT_PRECIP'].values[0] for ibasin_GID in basin_GID_1d_rst])
clima_PrecSeaInd = np.array([clima[clima['STAID']=='0%s'%(gageID_long.values[i])]['PRECIP_SEAS_IND'].values[0] for i in range(len(gageID_long))])
clima_PrecSeaInd_basin = np.array([clima[clima['STAID']==ibasin_GID]['PRECIP_SEAS_IND'].values[0] for ibasin_GID in basin_GID_1d_rst])

soils = pd.read_excel('%s/gagesII_sept30_2011_conterm.xlsx' % (rep_dir), sheetname='Soils', converters={'STAID': str})
soils_AWCAVE = np.array([soils[soils['STAID']=='0%s'%(gageID_long.values[i])]['AWCAVE'].values[0] for i in range(len(gageID_long))])
soils_AWCAVE_basin = np.array([soils[soils['STAID']==ibasin_GID]['AWCAVE'].values[0] for ibasin_GID in basin_GID_1d_rst])
soils_PERMAVE = np.array([soils[soils['STAID']=='0%s'%(gageID_long.values[i])]['PERMAVE'].values[0] for i in range(len(gageID_long))])
soils_PERMAVE_basin = np.array([soils[soils['STAID']==ibasin_GID]['PERMAVE'].values[0] for ibasin_GID in basin_GID_1d_rst])
soils_BDAVE = np.array([soils[soils['STAID']=='0%s'%(gageID_long.values[i])]['BDAVE'].values[0] for i in range(len(gageID_long))])
soils_OMAVE = np.array([soils[soils['STAID']=='0%s'%(gageID_long.values[i])]['OMAVE'].values[0] for i in range(len(gageID_long))])
soils_WTDEPAVE = np.array([soils[soils['STAID']=='0%s'%(gageID_long.values[i])]['WTDEPAVE'].values[0] for i in range(len(gageID_long))])
soils_WTDEPAVE_basin = np.array([soils[soils['STAID']==ibasin_GID]['WTDEPAVE'].values[0] for ibasin_GID in basin_GID_1d_rst])

morph = pd.read_excel('%s/gagesII_sept30_2011_conterm.xlsx' % (rep_dir), sheetname='Bas_Morph', converters={'STAID': str})
centroid_lat_HCDN = np.array([morph[morph['STAID']=='0%s'%(gageID_long.values[i])]['LAT_CENT'].values[0] for i in range(len(gageID_long))])
centroid_lon_HCDN = np.array([morph[morph['STAID']=='0%s'%(gageID_long.values[i])]['LONG_CENT'].values[0] for i in range(len(gageID_long))])
centroid_lat_basin = np.array([morph[morph['STAID']==ibasin_GID]['LAT_CENT'].values[0] for ibasin_GID in basin_GID_1d_rst])
centroid_lon_basin = np.array([morph[morph['STAID']==ibasin_GID]['LONG_CENT'].values[0] for ibasin_GID in basin_GID_1d_rst])

### Calculate the GEV parameters for the STATIONARY case
location = np.array([fitGEV('0%s' % (gageID_long.values[i]), stationary=True)[0] for i in range(gageID_long.shape[0])])
scale = np.array([fitGEV('0%s' % (gageID_long.values[i]), stationary=True)[1] for i in range(gageID_long.shape[0])])
shape = np.array([fitGEV('0%s' % (gageID_long.values[i]), stationary=True)[2] for i in range(gageID_long.shape[0])])

### Prepare the dataframe
gageInfo['mu'][gageID_long.index] = location
gageInfo['sigma'][gageID_long.index] = scale
gageInfo['xi'][gageID_long.index] = shape
gageInfo['DEM_mean'][gageID_long.index] = topo_MEAN
gageInfo['DEM_std'][gageID_long.index] = topo_std
gageInfo['SLP'][gageID_long.index] = topo_SLP
gageInfo['RR'][gageID_long.index] = topo_RR
gageInfo['TOPWET'][gageID_long.index] = hydro_TOPWET
gageInfo['BFI'][gageID_long.index] = hydro_BFI
gageInfo['streams'][gageID_long.index] = hydro_streams
gageInfo['sinuousity'][gageID_long.index] = hydro_sinuousity
gageInfo['contact'][gageID_long.index] = hydro_contact
gageInfo['SnowPct'][gageID_long.index] = clima_SnowPct
gageInfo['PrecSeaInd'][gageID_long.index] = clima_PrecSeaInd
gageInfo['AWCAVE'][gageID_long.index] = soils_AWCAVE
gageInfo['PERMAVE'][gageID_long.index] = soils_PERMAVE
gageInfo['BDAVE'][gageID_long.index] = soils_BDAVE
gageInfo['OMAVE'][gageID_long.index] = soils_OMAVE
gageInfo['WTDEPAVE'][gageID_long.index] = soils_WTDEPAVE
'''

'''
### Ordinary krigging
basin_GEV_loc_UK_pred = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, location)[0]
basin_GEV_sca_UK_pred = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, np.log10(scale))[0]    # Take the log to avoid negative values for kriging
basin_GEV_sca_UK_pred = np.power(10, basin_GEV_sca_UK_pred)
basin_GEV_shp_UK_pred = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, shape)[0]
basin_GEV_loc_UK_stdev = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, location)[1]
basin_GEV_sca_UK_stdev = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, np.log10(scale))[1]
basin_GEV_shp_UK_stdev = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, shape)[1]

basin_GEV_mu0_UK_pred = UK_basin(centroid_lat_HCDN, centroid_lon_HCDN, centroid_lat_basin, centroid_lon_basin, mu0)[0]

### Krigging with external drift
cov_obs_loc = np.column_stack([topo_MEAN, topo_std, hydro_BFI, soils_PERMAVE, soils_WTDEPAVE])
cov_pre_loc = np.column_stack([topo_MEAN_basin, topo_std_basin, hydro_BFI_basin, soils_PERMAVE_basin, soils_WTDEPAVE_basin])
cov_obs_sca = np.column_stack([soils_AWCAVE, hydro_TOPWET, hydro_BFI, clima_SnowPct, clima_PrecSeaInd])
cov_pre_sca = np.column_stack([soils_AWCAVE_basin, hydro_TOPWET_basin, hydro_BFI_basin, clima_SnowPct_basin, clima_PrecSeaInd_basin])
cov_obs_shp = np.column_stack([soils_AWCAVE, hydro_BFI, clima_PrecSeaInd, topo_MEAN, soils_PERMAVE])
cov_pre_shp = np.column_stack([soils_AWCAVE_basin, hydro_BFI_basin, clima_PrecSeaInd_basin, topo_MEAN_basin, soils_PERMAVE_basin])

basin_GEV_loc_KED_pred = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_loc, centroid_lat_basin, centroid_lon_basin, cov_pre_loc, location)[0]
basin_GEV_sca_KED_pred = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_sca, centroid_lat_basin, centroid_lon_basin, cov_pre_sca, np.log10(scale))[0]
basin_GEV_sca_KED_pred = np.power(10, basin_GEV_sca_KED_pred)
basin_GEV_shp_KED_pred = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_shp, centroid_lat_basin, centroid_lon_basin, cov_pre_shp, shape)[0]
basin_GEV_loc_KED_stdev = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_loc, centroid_lat_basin, centroid_lon_basin, cov_pre_loc, location)[1]
basin_GEV_sca_KED_stdev = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_sca, centroid_lat_basin, centroid_lon_basin, cov_pre_sca, np.log10(scale))[1]
basin_GEV_shp_KED_stdev = KED_basin(centroid_lat_HCDN, centroid_lon_HCDN, cov_obs_shp, centroid_lat_basin, centroid_lon_basin, cov_pre_shp, shape)[1]
'''

### Get the p-value from the TFPW test
#trend = np.array([TFPW_detect('0%s'%(gageID_long.values[i]))[0] for i in range(gageID_long.shape[0])])              ## trend from TFPW test for each station
#pvalue_trend = np.array([TFPW_detect('0%s'%(gageID_long.values[i]))[1] for i in range(gageID_long.shape[0])])       ## p-value from TFPW test for each station
#pvalue_mu1 = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), np.arange(gageStime[gageID_long.index].values[i],2015))[3] for i in range(gageID_long.shape[0])])
#pvalue_sigma1 = np.array([fitGEV_LRT('0%s' % (gageID_long.values[i]), np.arange(gageStime[gageID_long.index].values[i],2015))[4] for i in range(gageID_long.shape[0])])
#NSGEV(gageID_long)
