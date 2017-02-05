#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects.numpy2ri as rpyn

### Get the station ID
data_dir = '../HCDN/EastHghlnds'
fig_dir = '../Figures'
minLen = 60
gageInfo = pd.read_excel('%s/HCDN-2009_Station_Info_EastHghlnds.xlsx' % (data_dir))
gageID = gageInfo['STATION ID']
gageStime = gageInfo['stime (WY)']

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

def TFPW_detect(ID):   

    """
    Use the trend-free pre-whitening (TFPW) method to detect the trend
    """

    importr('zyp')
    data_valid = gage_max_flow[ID]
    data_valid = data_valid[~np.isnan(data_valid)]
    ts = FloatVector(data_valid)
    r.assign('ts', ts)
    result = r("result <- zyp.trend.vector(ts, method='yuepilon', conf.intervals=TRUE)")
    trend = result[1]
    pvalue = result[5]

    return trend, pvalue

'''
### Test the .dat file
for i in range(gageID.shape[0]):
    print gageID[i]
    get_annual_max(gageID[i], gageStime[i])
'''

gage_max_flow = {'0%s' % (gageID[i]): get_annual_max(gageID[i], gageStime[i]) for i in range(gageID.shape[0])}
gageValidLen = np.array([np.sum(~np.isnan(gage_max_flow['0%s' % (gageID[i])])) for i in range(gageID.shape[0])])
gageInfo['ValidLen'] = gageValidLen
index = gageInfo['ValidLen'] > minLen
gageID_long = gageID[index]
lat = gageInfo['LAT_GAGE'][index]
lon = gageInfo['LONG_GAGE'][index]

ps = np.array([TFPW_detect('0%s'%(gageID_long.values[i]))[1] for i in range(gageID_long.shape[0])])
ind = ps<=0.05
ind2 = ps>0.05
### Plot
plt.figure()
#M = Basemap(resolution='l', llcrnrlat=40, urcrnrlat=48, llcrnrlon=-81, urcrnrlon=-66)
M = Basemap(resolution='l', llcrnrlat=33, urcrnrlat=42, llcrnrlon=-95, urcrnrlon=-75)
M.scatter(lon[ind], lat[ind], color='k', s=150, marker='.')
M.scatter(lon[ind2], lat[ind2], color='r', s=150, marker='.')
M.drawcountries(linewidth=2)
M.drawcoastlines(linewidth=2)
M.drawstates()
M.etopo()
plt.savefig('%s/HCDN_sites_EastHghlnds_trend.pdf' % (fig_dir))
plt.show()


