#! /usr/local/bin/python

#-------------------------------------------------------------------------------
# PROG : HM01.py     <by Xiaogang He, hexg@princeton.edu>
# DESC : Python codes for the first assignment of CEE306 at Princeton University
# USAGE: $ python ./HM01.py 
#-------------------------------------------------------------------------------

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects.numpy2ri as rpyn

def read_peak_discharge(gageID):
    df = pd.read_csv('../Data/%s.txt' % (gageID), header=None, names=['Year', 'Month', 'Day', 'Discharge', 'xxx'])

    return df

### Read USGS observations
obs = read_peak_discharge('01401000')
year = obs['Year']
peak = obs['Discharge']

### Plot annual peak discharge observations
plt.figure()
plt.plot(peak)
plt.show()

### Use rpy2 to access R from within python
peak = FloatVector(peak)
r.assign('peak', peak)

### Test for trend in annual peak data
importr('Kendall')
print r("MannKendall(peak)")

### Test for change point in annual peak data
importr('strucchange')
r("bp.peak <- breakpoints(peak ~ 1)")
print r("bp.peak$breakpoints")

### Calculate the return period
def calculate_return_period(EVD, rp):

    importr('ismev')
    r.assign('rp', rp)

    if EVD == 'GEV':
        r("fit.obj <- gev.fit(peak)")
        r("pars <- fit.obj$mle")
        rl = r("gevq(fit.obj$mle, 1./rp)")[0]
        r.pdf(file='./%s_diag.pdf' % (EVD), width=8, height=8)
        r("gev.diag(fit.obj)")
        r("dev.off()")
    elif EVD == 'Gumbel':
        r("fit.obj <- gum.fit(peak)")
        r("pars <- fit.obj$mle")
        rl = r("gum.q(1.0/rp, pars[1], pars[2])")[0]
        r.pdf(file='./%s_diag.pdf' % (EVD), width=8, height=8)
        r("gum.diag(fit.obj)")
        r("dev.off()")

    return rl

rl_gev = calculate_return_period('GEV', 100)
rl_gum = calculate_return_period('Gumbel', 100)


